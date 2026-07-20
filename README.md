# Basic_Flow - Simple Voice Turn Detector & LLM Pipeline

A minimal, beginner-friendly implementation of the real-time voice pipeline:
**Speak into Mic ➔ ONNX Multimodal Model (Turn Detection) ➔ LLM Response (GPT-4o-mini)**

---

## 🚀 Quick Setup Guide (Step-by-Step)

Follow these simple steps to set up and run the project on any computer:

### Step 1: Open Terminal 
Open your terminal (PowerShell, Command Prompt, or Terminal) and navigate into this directory:
```bash
cd EOU
```

### Step 2: Set Up Python Virtual Environment
Create and activate a fresh virtual environment:

* **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **Linux / macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

---

### Step 3: Install Required Dependencies
Run this single command to install all necessary Python libraries:
```bash
pip install -r requirements.txt
```

---

### Step 4: Configure Your `.env` File
Create a file named `.env` in the `Basic_Flow` folder (or edit the existing `.env` file) and set the following two values:

```env
# 1. Your OpenAI API Key (Required for LLM responses)
OPENAI_API_KEY=your_actual_openai_api_key_here

# 2. Relative or absolute path to your trained ONNX model file (.onnx)
LOCAL_SMART_TURN_MODEL_PATH=model/model_int8_static_calib64.onnx

# 3. Sensitivity Threshold (Default: 0.30 works well for both Hindi & English)
# Adjust based on language preference (0.20-0.50):
# - 0.30: Recommended optimal value for Hindi & English (triggers complete thoughts, holds pauses)
# - 0.20: Faster response time for short phrases
# - 0.50: Conservative holding for slow speakers
SMART_TURN_THRESHOLD=0.30
```

> 💡 **Note on Model File**: Always use the **`.onnx`** file (e.g. `model/model_int8_static_calib64.onnx`), **NOT** the `.safetensors` file.

---

## 🏃 How to Run

Execute the main script:
```bash
python main.py
```

### How to Interact:
1. **Language Selection**: Type `1` for **Hindi** or `2` for **English** and press Enter.
2. **Speak**: Talk into your microphone naturally.
3. **Pause**: Pause for ~1 second when you finish speaking.
4. **ONNX Decision**:
   * If the model detects your sentence is **COMPLETE**, it triggers GPT-4o-mini and prints the AI response.
   * If the model detects an **INCOMPLETE** pause (e.g. you paused mid-thought), it holds and waits for you to continue.

