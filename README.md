# Basic_Flow - Simple Speak -> ONNX -> LLM Pipeline

This folder contains a minimal, zero-fluff pipeline implementing the basic flow:
**Speak (Microphone) ➔ Multimodal ONNX Turn Detector ➔ OpenAI GPT Brain ➔ Console Response**

---

## 1. Setup

1. **Activate the environment**:
   ```powershell
   .\testing\Scripts\activate
   ```

2. **Configure credentials**:
   Check `.env` in this folder:
   * `OPENAI_API_KEY`: OpenAI API Key to fetch the GPT-4o-mini response.
   * `LOCAL_SMART_TURN_MODEL_PATH`: Set to the absolute path of your trained **ONNX model** file.

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 2. Running the Flow

Start the script:
```powershell
.\testing\Scripts\python.exe Basic_Flow/main.py
```

1. Select **1** for Hindi (hi-IN) or **2** for English (en-IN).
2. Speak clearly.
3. Pause for a second to let the microphone trigger a silence detection point.
4. The ONNX model will decide if you finished your sentence or if you were just thinking.
5. If **complete**, it will call GPT-4o-mini and print the agent's response. If **incomplete**, it will hold and wait for you to keep speaking.
