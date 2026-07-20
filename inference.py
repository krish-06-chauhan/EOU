import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, AutoTokenizer
import os
import glob
from dotenv import load_dotenv

from audio_utils import truncate_audio_to_last_n_seconds

base_dir = os.path.dirname(os.path.abspath(__file__))
# Load environment variables from .env in Basic_Flow directory
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))
env_path = os.environ.get("LOCAL_SMART_TURN_MODEL_PATH", "model/model_int8_static_calib64.onnx")

if os.path.isabs(env_path):
    ONNX_MODEL_PATH = env_path
else:
    ONNX_MODEL_PATH = os.path.abspath(os.path.join(base_dir, env_path))

if not os.path.exists(ONNX_MODEL_PATH):
    raise FileNotFoundError(
        f"ONNX model file not found at: '{ONNX_MODEL_PATH}'. "
        "Please configure LOCAL_SMART_TURN_MODEL_PATH in .env or place your .onnx model in the Basic_Flow/model/ folder."
    )

print(f"[SmartTurn] Loading ONNX model from: {ONNX_MODEL_PATH}")

def build_session(onnx_path):
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so)

# Initialize models
text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
feature_extractor = WhisperFeatureExtractor(chunk_length=8)
session = build_session(ONNX_MODEL_PATH)


def predict_endpoint(audio_array, silence_duration=0.0, trailing_text=""):
    """
    Predict complete (1) vs incomplete (0) thought.
    """
    # Append silence_duration seconds of zeros to the end to align with training distribution
    silence_samples = int(silence_duration * 16000)
    if silence_samples > 0:
        audio_array = np.concatenate([audio_array, np.zeros(silence_samples, dtype=np.float32)])

    # Pad/truncate to 8 seconds
    audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

    # Whisper acoustic features
    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    )
    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)

    # Tokenize text
    t_inputs = text_tokenizer(
        trailing_text,
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="np"
    )
    text_input_ids = t_inputs.input_ids.astype(np.int64)
    text_attention_mask = t_inputs.attention_mask.astype(np.int64)
    silence_dur_tensor = np.array([[float(silence_duration)]], dtype=np.float32)

    # Run ONNX session with dynamic shapes mapping
    model_inputs = [x.name for x in session.get_inputs()]
    feed_dict = {"input_features": input_features}
    
    if "silence_duration" in model_inputs:
        feed_dict["silence_duration"] = silence_dur_tensor
    if "text_input_ids" in model_inputs:
        feed_dict["text_input_ids"] = text_input_ids
    if "text_attention_mask" in model_inputs:
        feed_dict["text_attention_mask"] = text_attention_mask

    outputs = session.run(None, feed_dict)
    probability = outputs[0][0].item()
    prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": probability,
    }
