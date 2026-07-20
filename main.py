import os
import sys
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import speech_recognition as sr

from inference import predict_endpoint

# Force UTF-8 terminal encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

base_dir = os.path.dirname(os.path.abspath(__file__))
# Load keys explicitly from local .env
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None


def merge_transcript(existing_buffer: str, new_chunk: str) -> str:
    """
    Robustly merges a new STT chunk into the existing transcript buffer,
    eliminating STT duplicate loops while preserving new words and protecting against data loss.
    """
    existing_clean = existing_buffer.strip()
    new_clean = new_chunk.strip()

    if not existing_clean:
        return new_clean
    if not new_clean or new_clean == existing_clean:
        return existing_clean

    words_existing = existing_clean.split()
    words_new = new_clean.split()

    # Case A: new_chunk completely subsumes or extends existing_buffer from the start
    if new_clean.startswith(existing_clean):
        return new_clean

    # Case B: existing_buffer already ends with new_chunk
    if existing_clean.endswith(new_clean):
        return existing_clean

    # Case C: Find maximum overlapping suffix of existing_buffer matching a prefix of new_chunk
    max_possible_overlap = min(len(words_existing), len(words_new))
    best_overlap_k = 0

    for k in range(max_possible_overlap, 0, -1):
        if words_existing[-k:] == words_new[:k]:
            best_overlap_k = k
            break

    if best_overlap_k > 0:
        new_words = words_new[best_overlap_k:]
        merged = " ".join(words_existing + new_words)
    else:
        if new_clean in existing_clean:
            return existing_clean
        merged = f"{existing_clean} {new_clean}"

    # Invariant: Never allow buffer to shrink
    if len(merged.split()) < len(words_existing):
        return existing_clean

    return merged


def get_llm_response(prompt_text):
    """Call OpenAI GPT-4o-mini to get a quick response."""
    if not client:
        return f"[Mock LLM Response] Received: '{prompt_text}'. Please configure OPENAI_API_KEY to get real responses."
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a direct, short voice assistant. Respond in the same language as the user in 1 sentence."},
                {"role": "user", "content": prompt_text}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Call failed: {e}"


def run_basic_flow():
    print("\n" + "="*60)
    print("   BASIC FLOW: SPEAK -> ONNX DETECTOR -> LLM RESPONSE")
    print("="*60)
    print("Instructions: Speak into your microphone. Pause for a second to trigger detection.")
    print("The ONNX model will determine if you are done speaking before triggering the LLM.")
    print("Press Ctrl+C to exit.\n")

    # Set language: hi-IN (Hindi) or en-IN (English)
    print("Choose language:")
    print("  1. Hindi (hi-IN)")
    print("  2. English (en-IN)")
    lang_choice = input("Enter choice (1/2, default 1): ").strip()
    stt_lang = "en-IN" if lang_choice == "2" else "hi-IN"
    print(f"-> Language set to: {stt_lang}\n")

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8  # Silence delay (seconds) to trigger EOU evaluation
    microphone = sr.Microphone(sample_rate=16000)

    # Ambient noise calibration
    with microphone as source:
        print("🔊 Calibrating microphone for background noise (1s)...")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        print("✅ Microphone calibrated. Start speaking!")

    SILENCE_TIMEOUT = 1.0  # VAD pause length (seconds) passed to the ONNX model
    buffer_transcript = ""

    while True:
        print("\n🎤 Listening...")
        with microphone as source:
            try:
                # Capture speech chunk
                audio_data = recognizer.listen(source, timeout=SILENCE_TIMEOUT, phrase_time_limit=10.0)
            except sr.WaitTimeoutError:
                # User paused speaking completely
                continue
            except KeyboardInterrupt:
                print("\nExiting basic flow...")
                break

        print("⚡ Transcribing audio...")
        try:
            chunk_text = recognizer.recognize_google(audio_data, language=stt_lang).strip()
        except sr.UnknownValueError:
            print("❌ Unintelligible sound. Speak clearly.")
            continue
        except sr.RequestError as e:
            print(f"STT Error: {e}")
            break

        # Merge transcript text cleanly using suffix/prefix overlap matching
        buffer_transcript = merge_transcript(buffer_transcript, chunk_text)

        print(f"➔ [Current Transcript Buffer]: \"{buffer_transcript}\"")

        # Convert recorded PCM audio bytes to float32 NumPy array at 16000 Hz sample rate
        raw_pcm_bytes = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
        int16_samples = np.frombuffer(raw_pcm_bytes, dtype=np.int16)
        audio_f32 = int16_samples.astype(np.float32) / 32768.0

        # Run ONNX Model Evaluation
        threshold = float(os.getenv("SMART_TURN_THRESHOLD", "0.50"))
        print(f"🧠 Evaluating turn completion with ONNX model (Threshold = {threshold:.2f})...")
        res = predict_endpoint(
            audio_array=audio_f32,
            silence_duration=SILENCE_TIMEOUT,
            trailing_text=buffer_transcript
        )
        
        probability = res["probability"]
        is_complete = probability >= threshold

        if is_complete:
            print(f"🟩 [ONNX DECISION]: COMPLETE (p={probability:.4f}) -> Triggering LLM...")
            # Fetch reply
            llm_reply = get_llm_response(buffer_transcript)
            print(f"🤖 LLM Response: \"{llm_reply}\"")
            # Clear buffer for the next turn
            buffer_transcript = ""
            print("-"*50)
        else:
            print(f"🟨 [ONNX DECISION]: INCOMPLETE (p={probability:.4f}) -> Holding turn, please continue...")


if __name__ == "__main__":
    try:
        run_basic_flow()
    except KeyboardInterrupt:
        print("\nGoodbye!")
