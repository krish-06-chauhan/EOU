# Technical Architecture & Deep-Dive Specification

This document provides a low-level engineering breakdown of the **Multimodal Turn Detection System** designed for high-level technical discussions (e.g., with CTOs, System Architects, and Lead ML Engineers).

---

## 1. Executive Summary & Problem Definition

Standard VAD (Voice Activity Detection) and audio-only End-of-Utterance (EOU) models suffer from high **False Cutoff Rates (~49.5%)**. When a user pauses mid-thought (e.g., *"I want to order a pizza... [thinking]... with extra cheese"*), acoustic-only models interpret the silence as turn completion and prematurely trigger the LLM.

To solve this, we engineered a **Multimodal Fusion Architecture** combining **Acoustic Features**, **Silence Timing**, and **Multilingual Lexical Semantics**.

---

## 2. Neural Network Architecture Breakdown

The model fuses three distinct input representations into an 800-dimensional unified embedding space before classification:

```
                               ┌────────────────────────────────────────┐
   Log-Mel Spectrogram (80x800)│ Whisper-tiny Encoder (Frozen Backbone) │──► Pooled Acoustic (384-dim)
                               └────────────────────────────────────────┘
                                                                             │
                               ┌────────────────────────────────────────┐    │
   Silence Duration (float32)  │  Linear(1,32) ──► GELU ──► Linear(32,32)│──► Silence Projection (32-dim)  ──► [ Concatenation (800-dim) ] ──► FFN Classifier ──► Sigmoid P(Complete)
                               └────────────────────────────────────────┘    │
                                                                             │
                               ┌────────────────────────────────────────┐    │
   Trailing Words (Tokens)     │ Multilingual MiniLM-L12-v2 + Mask Pooling│──► Pooled Lexical (384-dim)
                               └────────────────────────────────────────┘
```

### A. Acoustic Branch (384-dim)
* **Input**: 8-second 16kHz log-mel spectrogram tensor `[batch, 80, 800]`.
* **Backbone**: Frozen HuggingFace `Whisper-tiny` Encoder.
* **Attention Pooling**: Uses a learned linear projection layer with softmax weights over time steps (`pool_attention`) to produce a fixed 384-dimensional acoustic summary vector.

### B. Silence Timing Branch (32-dim)
* **Input**: Tensor `[batch, 1]` representing the exact elapsed silence duration in seconds.
* **Projection**: Non-linear feed-forward network (`Linear(1, 32) -> GELU -> Linear(32, 32)`).
* **Rationale**: Gives the classification head explicit quantitative awareness of pause duration.

### C. Lexical Semantic Branch (384-dim)
* **Input**: Tokens (`input_ids` `[batch, 32]`, `attention_mask` `[batch, 32]`) for the trailing 15 spoken words transcript leading up to the pause.
* **Backbone**: Frozen `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
* **Masked Mean Pooling**: To prevent zero/padding `<pad>` tokens from corrupting sentence embeddings, we compute masked mean pooling:
  $$\text{Pooled} = \frac{\sum (\text{Hidden\_States} \times \text{Mask})}{\max(\sum \text{Mask}, 1e-9)}$$

### D. Classification Head
* Concatenates the representations: $[384 + 32 + 384] = 800$ dimensions.
* Passes through a multi-layer perceptual head returning a single logit, converted via Sigmoid to a probability $p \in [0.0, 1.0]$.
  * $p > 0.5 \implies$ **Turn Complete (Trigger LLM)**
  * $p \le 0.5 \implies$ **Turn Incomplete (Hold / Keep Listening)**

---

## 3. Critical Low-Level Engineering & Pipeline Fixes

### A. Microphone Sample Rate Resampling Normalization
* **Problem**: Hardware soundcards record natively at 44.1 kHz or 48 kHz. Feeding un-resampled PCM data into Whisper (which expects 16 kHz) pitch-shifts and time-stretches audio by 2.7x, causing model failure.
* **Fix**: Enforced exact 16,000 Hz 16-bit mono PCM conversion using SpeechRecognition's native resampler:
  ```python
  raw_pcm_bytes = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
  ```

### B. Training/Inference Distribution Alignment (Silence Padding)
* **Problem**: Training clips contain $N$ seconds of physical silence at the end of the audio wave. Real-time VAD cuts recording immediately when speech stops (0 trailing silence).
* **Fix**: Synthesize and append $N$ seconds of zero-padding matching `silence_duration` to the audio array prior to 8-second window truncation:
  ```python
  silence_samples = int(silence_duration * 16000)
  if silence_samples > 0:
      audio_array = np.concatenate([audio_array, np.zeros(silence_samples, dtype=np.float32)])
  ```

### C. INT8 Static QDQ Quantization & Optimization
* Model exported to ONNX FP32 and statically quantized using **Entropy Calibration (QDQ format, UINT8 activations, INT8 weights)** over training data batches.
* **Result**: Reduced memory footprint by ~60%, achieving **sub-160ms CPU latency** while maintaining **95.91% accuracy** (only 0.52% degradation vs FP32).

---

## 4. Key Metrics Summary

| Metric | Metric Value | CTO / Business Significance |
| :--- | :---: | :--- |
| **Overall Accuracy** | **95.91%** | Evaluated across 2,324 speech-turn evaluation points. |
| **English Accuracy** | **97.44%** | Exceptional performance on English turns. |
| **Hindi Accuracy** | **94.13%** | Native support for Hindi & Hinglish code-switching. |
| **False Positive Rate (FPR)** | **1.68%** | **False Cutoff Rate**: The agent will only interrupt the user **1.68% of the time**. |
| **False Negative Rate (FNR)** | **2.41%** | **False Delay Rate**: The agent fails to respond when expected only **2.41% of the time**. |
| **CPU End-to-End Latency** | **~156.8 ms** | Total processing latency on standard CPU (Feature extraction + ONNX inference). |
