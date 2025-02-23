# Video Processing AI (Offline)

This project is an **AI-powered video processing tool** that works **completely offline** on a personal computer. It can:
- **Detect objects** in videos using YOLO.
- **Generate captions** for images using BLIP.
- **Transcribe speech** from audio using Whisper.
- **Detect sounds/events** using PANNs.
- **Summarize videos** using LLMs from **Ollama**.

It is designed as a **home research project**, showing that **powerful AI analysis** can run locally without fine-tuning or retraining.

---

## 🔧 Installation

### 1. Install Dependencies

#### Using Pip:
```bash
pip install -r pip_requirements.txt
```

#### Using Conda:
```bash
conda install --file conda_requirements.txt
```

### 2. Install Ollama
Ollama is required for AI-powered text summarization. Install it using:

#### On Linux / macOS:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### On Windows (PowerShell):
```powershell
iwr -useb https://ollama.com/install.ps1 | iex
```

### 3. Download Required Models

#### Ollama LLM Models:
```bash
ollama pull phi4:latest
ollama pull partai/dorna-llama3:latest
ollama pull qwen:14b
```

#### YOLO Model:
- **Download from:** [YOLOv8 weights](https://github.com/ultralytics/ultralytics)
- **Save in project directory.**

#### Whisper Model:
```python
import whisper
whisper.load_model("large-v2")
```

#### BLIP Model:
BLIP models are downloaded **automatically** when used.

#### PANNs Model:
- **Download from:** [PANNs GitHub](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- **Save as:** `models/cnn14.pth`

### 4. Set Up Environment (Windows Users)
- Install **Tesseract OCR** from: [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- Set its path in `pytesseract.pytesseract.tesseract_cmd`.

---

## 🚀 Usage

### Run the GUI
```bash
python video_processing_gui.py
```
- Select a **video file**.
- Choose **language settings** for transcription.
- Choose an **AI summarization model**.
- Click **Start Processing**.

### Run Without GUI (Command Line)
```bash
python video_processing.py --video path/to/video.mp4 --save
```

---

## 📂 Project Structure

Expected directory structure:

```
{dir_structure_str}
```

---

## ⚡ System Requirements
- **Recommended GPU:** NVIDIA RTX 3060 / 3070 or higher
- **RAM:** Minimum **16GB**, recommended **32GB**
- **Disk Space:** At least **20GB** free for models

---

## 🔍 How It Works

1. **Extracts audio** from the video.
2. **Transcribes speech** using Whisper.
3. **Detects objects** using YOLO.
4. **Generates captions** using BLIP.
5. **Detects sound events** using PANNs.
6. **Summarizes video** using LLM (via Ollama).
7. **Generates final annotated video and report**.

---

## 🛠 Troubleshooting

### Issue: "CUDA out of memory"
- Use a **lower memory model** or **reduce batch size**.
- Close other GPU-heavy applications.
- Try running on **CPU** (slower but works).

### Issue: "Ollama not found"
- Ensure Ollama is installed and added to `PATH`.
- Try reinstalling:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

---

## 🤝 Contributing
This project is **open-source**. Feel free to **fork**, **improve**, or **suggest enhancements**.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgments
- **Ultralytics** (YOLO models)
- **OpenAI** (Whisper)
- **Salesforce** (BLIP)
- **Hugging Face** (Model Hosting)
- **Ollama** (On-device LLMs)

