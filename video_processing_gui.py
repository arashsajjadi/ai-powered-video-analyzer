#!/usr/bin/env python
# video_processing_gui.py

import os
import re
import math
import cv2
import logging
import platform
import psutil
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract
import whisper
from panns_inference import AudioTagging, labels as pann_labels
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import subprocess
import shutil
import warnings
import time
import base64
import ollama  # Requires the ollama Python package

# --- Dynamic Path Setup (for Dockerization / cross-platform) ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # Set TESSDATA_PREFIX to the parent directory containing tessdata (not used anymore)
    os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR"
    PANN_MODEL_PATH = r"C:\Users\arash\panns_data\cnn14.pth"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    PANN_MODEL_PATH = "/app/models/cnn14.pth"

# --- Suppress extraneous warnings ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Setup Logging ---
LOG_FILE = "video_processing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# --- YOLO Class Mapping ---
CLASS_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "brocolli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush"
}

# --- Global Constants & Variables ---
LLAVA_INTERVAL = 5  # LLava is removed for speed

# --- Helper: Convert Seconds to HH:MM:SS ---
def seconds_to_timestr(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

# --- Hardware Usage Debug Print ---
def print_hardware_usage():
    print("=== Hardware Usage ===")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    mem = psutil.virtual_memory()
    print(f"Memory Usage: {mem.used / (1024 ** 3):.1f} GB / {mem.total / (1024 ** 3):.1f} GB ({mem.percent}%)")
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
    print("======================\n")

# --- Audio Preprocessing ---
def preprocess_audio(audio_file, sr=32000):
    waveform, sr = librosa.load(audio_file, sr=sr)
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    return waveform, sr

# --- Utility Functions ---
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError("No audio track found in the video.")
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.reader.close()
    clip.audio.reader.close_proc()

def transcribe_audio(audio_file, language=None):
    waveform, sr = preprocess_audio(audio_file)
    proc_audio = "temp_proc_audio.wav"
    sf.write(proc_audio, waveform, sr)
    if language == "fas":
        language = "fa"
    try:
        result = whisper_model.transcribe(proc_audio, task="transcribe", language=language)
        detected_language = result.get("language", "unknown")
        logging.info("Detected audio language: %s", detected_language)
        transcription = result["text"]
    except Exception as e:
        logging.error("Error in audio transcription: %s", str(e))
        transcription = ""
        detected_language = "unknown"
    os.remove(proc_audio)
    if detected_language.lower() == "unknown" or not detected_language:
        detected_language = "eng"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return transcription, detected_language

def detect_audio_events(audio_file):
    try:
        waveform, sr = librosa.load(audio_file, sr=32000)
        segment_length = 5 * sr  # 5-second segments
        events = {}
        for i in range(0, len(waveform), segment_length):
            segment = waveform[i:i + segment_length]
            if len(segment) == 0:
                continue
            segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
            output = panns_model.inference(segment_tensor)
            if isinstance(output, dict) and "clipwise_output" in output:
                clipwise_output = np.array(output["clipwise_output"], dtype=float)
            else:
                clipwise_output = np.array(output, dtype=float)
            if np.max(clipwise_output) < 0.1:
                continue
            top_idx = int(np.argmax(clipwise_output))
            event_label = pann_labels[top_idx] if top_idx < len(pann_labels) else "Unknown"
            timestamp = i / sr
            if event_label in events:
                events[event_label].append(seconds_to_timestr(timestamp))
            else:
                events[event_label] = [seconds_to_timestr(timestamp)]
        if not events:
            return {"No event": []}
        return events
    except Exception as e:
        logging.error("Error in audio event detection: %s", str(e))
        return {"Error": []}

# --- (OCR functionality removed completely) ---

def clean_report(text):
    text = re.sub(r'[\u06F0-\u06F9]+', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text

def free_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Helper to describe detection position ---
def describe_position(x_norm, y_norm):
    if x_norm < 0.33:
        horz = "left"
    elif x_norm < 0.66:
        horz = "center"
    else:
        horz = "right"
    if y_norm < 0.33:
        vert = "top"
    elif y_norm < 0.66:
        vert = "middle"
    else:
        vert = "bottom"
    return f"{horz}, {vert}"

def article_for(label):
    return "an" if label[0].lower() in "aeiou" else "a"

# --- Function to get available Ollama models dynamically ---
def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            logging.error("Error calling 'ollama list': %s", result.stderr)
            return []
        lines = result.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception as e:
        logging.error("Exception in get_ollama_models: %s", str(e))
        return []

# --- Global Model Loading (with GPU memory cleanup after each load) ---
logging.info("Loading Whisper model (Large-v2)...")
whisper_model = whisper.load_model("large-v2")
free_gpu_memory()

logging.info("Loading PANNs audio detection model...")
panns_model = AudioTagging(checkpoint_path=PANN_MODEL_PATH)
free_gpu_memory()
print_hardware_usage()

# --- Modified LLM Integration Functions ---
def call_ollama(prompt, input_text, model):
    try:
        combined = prompt + "\n\n" + input_text
        result = subprocess.run(
            ["ollama", "run", model],
            input=combined,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        if result.returncode != 0:
            logging.error("Ollama call failed with code %d: %s", result.returncode, result.stderr)
            return "LLM call failed."
        output = re.sub(r'<think>.*?</think>', '', result.stdout, flags=re.DOTALL).strip()
        if not output:
            output = "No response received."
        print(f"LLM ({model}) output:\n{output}\n")
        return output
    except Exception as e:
        logging.error("Error calling Ollama: %s", str(e))
        return "Error in LLM call."

def ollama_summarize_report(report_file, model):
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report_text = f.read()
    except Exception as e:
        logging.error("Error reading report file: %s", str(e))
        return ""
    clean_text = clean_report(report_text)
    prompt = (
        '''
        You are an expert video content summarizer. Generate a cohesive, engaging, and concise narrative summary (less than 100 words) of the video based on the following report. Do not include timestamps, technical details, or model names—write in plain, natural language.
        '''
    )
    summary = call_ollama(prompt, clean_text, model=model)
    if "LLM call failed" in summary or "Error in LLM call" in summary:
        summary = "The video presents a dynamic scene with various events, blending spoken words and visuals into an engaging narrative."
    return summary

def generate_video_description():
    report_file = "report.txt"
    if not os.path.exists(report_file):
        logging.error("Report file not found for summarization.")
        return None
    summary = ollama_summarize_report(report_file, model=selected_summarization_model.get())
    description_file = "video_description.txt"
    try:
        with open(description_file, "w", encoding="utf-8") as f:
            f.write("Video Narrative Summary:\n")
            f.write(summary)
        logging.info("Video description generated as %s", description_file)
    except Exception as e:
        logging.error("Error generating video description: %s", str(e))
        return None
    return description_file

# --- GPU Memory Cleanup ---
def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Main Video Processing Function ---
def process_video(video_path, sample_rate=1, draw_boxes=True, save_video=False, show_video=False, ocr_languages="eng"):
    if not os.path.exists(video_path):
        logging.error("File does not exist: %s", video_path)
        return

    # --- Video Properties ---
    clip = VideoFileClip(video_path)
    fps = clip.fps
    frame_count = int(clip.reader.nframes)
    duration = clip.duration
    width, height = clip.size
    video_format = os.path.splitext(video_path)[1].lower()
    clip.reader.close()
    if clip.audio is not None:
        clip.audio.reader.close_proc()

    logging.info("Video properties: Duration: %s, Frames: %d, FPS: %.2f, Resolution: %dx%d, Format: %s",
                 seconds_to_timestr(duration), frame_count, fps, width, height, video_format)

    # --- Build Report Header ---
    report_lines = []
    report_lines.append("Video Processing Report")
    report_lines.append("-----------------------")
    report_lines.append(f"File: {video_path}")
    report_lines.append(f"Duration: {seconds_to_timestr(duration)} ({duration:.2f} seconds)")
    report_lines.append(f"Resolution: {width}x{height}")
    report_lines.append(f"FPS: {fps:.2f}")
    report_lines.append(f"Total Frames: {frame_count}")
    report_lines.append(f"Format: {video_format}")
    report_lines.append("")

    # --- Audio Analysis ---
    temp_audio_path = "temp_audio.wav"
    logging.info("Extracting audio from video...")
    try:
        extract_audio(video_path, temp_audio_path)
    except Exception as e:
        logging.error("Error extracting audio: %s", str(e))
        temp_audio_path = None

    audio_transcript = ""
    detected_lang = "unknown"
    audio_for_video = "audio_for_video.wav"
    if temp_audio_path and os.path.exists(temp_audio_path):
        shutil.copy(temp_audio_path, audio_for_video)
        try:
            logging.info("Transcribing audio with Whisper...")
            audio_transcript, detected_lang = transcribe_audio(temp_audio_path)
            logging.info("Audio transcription: %s", audio_transcript)
            logging.info("Detected language for transcription: %s", detected_lang)
        except Exception as e:
            logging.error("Error in audio transcription: %s", str(e))
        try:
            logging.info("Detecting audio events using PANNs (5-sec segments)...")
            audio_events = detect_audio_events(temp_audio_path)
            logging.info("Detected audio events: %s", audio_events)
        except Exception as e:
            logging.error("Error in audio event detection: %s", str(e))
            audio_events = {"Error": []}
        os.remove(temp_audio_path)
    else:
        logging.info("No audio extracted.")
        audio_events = {"No audio": []}

    report_lines.append("Audio Analysis:")
    report_lines.append(f"  Transcription: {audio_transcript if audio_transcript else 'N/A'}")
    if isinstance(audio_events, dict):
        for event, times in audio_events.items():
            times_str = ", ".join(times) if times else "N/A"
            report_lines.append(f"  Detected Audio Event - {event}: {times_str}")
    else:
        report_lines.append(f"  Detected Audio Event: {audio_events if audio_events else 'N/A'}")
    report_lines.append("")

    print_hardware_usage()
    free_gpu()

    # --- Load Advanced YOLO and BLIP Models (and free GPU memory after each load) ---
    logging.info("Loading advanced YOLO model (YOLO11x)...")
    try:
        yolo_model = YOLO("yolo11x.pt")
        free_gpu()
    except Exception as e:
        logging.error("Error loading YOLO model: %s", str(e))
        return

    logging.info("Loading BLIP-2 captioning model (base variant)...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.to(device)
        free_gpu()
    except Exception as e:
        logging.error("Error loading BLIP model: %s", str(e))
        return

    # --- Prepare Annotated Video Writer if needed ---
    annotated_temp = "annotated_temp.mp4"
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_temp, fourcc, fps, (width, height))

    # --- Process Video Frames ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error: Could not open video file.")
        return

    processed_frame_count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_rate != 0:
            if save_video and writer is not None:
                writer.write(frame)
            continue

        processed_frame_count += 1
        current_time = frame_idx / fps
        time_str = seconds_to_timestr(current_time)

        # --- YOLO Detection ---
        results = yolo_model(frame)
        yolo_descriptions = []
        for result in results:
            if result.boxes is None or result.boxes.data is None:
                continue
            detections = result.boxes.data.cpu().numpy()
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cls_int = int(cls)
                label = CLASS_MAP.get(cls_int, f"Unknown")
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                cx_norm = cx / width
                cy_norm = cy / height
                pos_descr = describe_position(cx_norm, cy_norm)
                phrase = f"{article_for(label)} {label} at {pos_descr}"
                yolo_descriptions.append(phrase)
        yolo_text = ", ".join(yolo_descriptions) if yolo_descriptions else None

        # --- BLIP Captioning ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        inputs = blip_processor(pil_img, return_tensors="pt").to(device)
        try:
            output_ids = blip_model.generate(
                **inputs,
                max_length=60,  # Allows more words (default is ~30)
                min_length=20,  # Forces captions to be at least 20 tokens long
                repetition_penalty=1.05,  # Reduces word repetition (higher value = less repetition)
                num_beams=5,  # Beam search improves caption quality (default is 1)
                length_penalty=0.6  # Encourages longer, more detailed captions
            )

            caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)

            # Remove adjacent repeated words from caption
            words = caption.split()
            if words:
                clean_words = [words[0]]
                for w in words[1:]:
                    if w.lower() != clean_words[-1].lower():
                        clean_words.append(w)
                caption = " ".join(clean_words)
            caption = caption if caption.strip() != "" else None
        except Exception as e:
            logging.error("Error generating BLIP caption: %s", str(e))
            caption = None

        # --- Build Log Line (only include non-N/A fields in plain language) ---
        log_fields = []
        if yolo_text:
            log_fields.append(yolo_text)
        if caption:
            log_fields.append(caption)
        log_line = f"Time {time_str}: " + " | ".join(log_fields)
        logging.info(log_line)
        report_lines.append(log_line)

        # --- Optionally, draw BLIP caption overlay on the frame ---
        if draw_boxes and caption:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_color = (255, 255, 255)
            bg_color = (0, 0, 0)
            margin = 10
            (txt_w, txt_h), baseline = cv2.getTextSize(caption, font, font_scale, thickness)
            x_txt = int((width - txt_w) / 2)
            y_txt = height - margin
            cv2.rectangle(frame,
                          (x_txt - margin, y_txt - txt_h - margin),
                          (x_txt + txt_w + margin, y_txt + baseline + margin),
                          bg_color,
                          cv2.FILLED)
            cv2.putText(frame, caption, (x_txt, y_txt), font, font_scale, text_color, thickness, cv2.LINE_AA)

        if save_video and writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    logging.info("Finished processing video.")
    free_gpu()

    # --- Merge Annotated Video with Audio ---
    if save_video and os.path.exists(audio_for_video):
        try:
            video_clip = VideoFileClip(annotated_temp)
            audio_clip = AudioFileClip(audio_for_video)
            video_with_audio = video_clip.set_audio(audio_clip)
            final_video = os.path.splitext(video_path)[0] + "_annotated.mp4"
            video_with_audio.write_videofile(final_video, codec="libx264", audio_codec="aac")
            logging.info("Annotated video with audio saved as %s", final_video)
            os.remove(annotated_temp)
            os.remove(audio_for_video)
        except Exception as e:
            logging.error("Error merging audio with annotated video: %s", str(e))
            final_video = annotated_temp
            logging.info("Annotated video saved without audio as %s", final_video)

    # --- Write Final Report ---
    report_filename = "report.txt"
    try:
        with open(report_filename, "w", encoding="utf-8") as rpt:
            rpt.write("\n".join(report_lines))
        logging.info("Report generated as %s", report_filename)
    except Exception as e:
        logging.error("Error generating report: %s", str(e))

    logging.info("Final Audio Transcription: %s", audio_transcript)
    logging.info("Final Detected Audio Events: %s", audio_events)

    # --- Ollama Summarization Feature (using user-selected model) ---
    desc_file = generate_video_description()
    if desc_file:
        logging.info("Video description generated: %s", desc_file)
    free_gpu()


# --- GUI Code ---
class VideoProcessingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Video Processing GUI")
        master.configure(bg="#f0f0f0")  # Light gray background

        # Row 0: Video File Selection
        self.video_path = tk.StringVar()
        tk.Label(master, text="Video File:", bg="#f0f0f0", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.video_entry = tk.Entry(master, textvariable=self.video_path, width=50, font=("Helvetica", 10))
        self.video_entry.grid(row=0, column=1, padx=5, pady=5)
        self.load_button = tk.Button(master, text="Load Video", command=self.load_video, font=("Helvetica", 10))
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        # Row 1: Language Options for Transcription
        self.auto_lang = tk.BooleanVar(value=True)
        self.auto_check = tk.Checkbutton(master, text="Auto Detect Language (for transcription)", variable=self.auto_lang, command=self.toggle_language_options, font=("Helvetica", 10), bg="#f0f0f0")
        self.auto_check.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        tk.Label(master, text="Primary Language:", bg="#f0f0f0", font=("Helvetica", 10)).grid(row=2, column=0, padx=5, pady=5, sticky="e")
        tk.Label(master, text="Secondary Language:", bg="#f0f0f0", font=("Helvetica", 10)).grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.languages = [
            ("English", "eng"),
            ("Persian", "fas"),
            ("Spanish", "spa"),
            ("French", "fra"),
            ("German", "deu"),
            ("Arabic", "ara"),
            ("Simplified Chinese", "chi_sim"),
            ("Traditional Chinese", "chi_tra"),
            ("Italian", "ita"),
            ("Japanese", "jpn"),
            ("Korean", "kor"),
            ("Russian", "rus"),
            ("None", "none")
        ]
        lang_names = [f"{name} ({code})" for name, code in self.languages]
        self.primary_lang = tk.StringVar(value=lang_names[0])
        self.secondary_lang = tk.StringVar(value=lang_names[-1])
        self.primary_menu = ttk.Combobox(master, textvariable=self.primary_lang, values=lang_names, state="readonly", font=("Helvetica", 10))
        self.primary_menu.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.secondary_menu = ttk.Combobox(master, textvariable=self.secondary_lang, values=lang_names, state="readonly", font=("Helvetica", 10))
        self.secondary_menu.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Row 4: Summarization Model Selection (Dropdown - dynamic list)
        tk.Label(master, text="Summarization Model:", bg="#f0f0f0", font=("Helvetica", 10, "bold")).grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.available_models = get_ollama_models()
        self.selected_model = tk.StringVar(value=self.available_models[0] if self.available_models else "N/A")
        self.model_menu = ttk.Combobox(master, textvariable=self.selected_model, values=self.available_models, state="readonly", font=("Helvetica", 10))
        self.model_menu.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Row 5: Options for saving and showing video
        self.save_video = tk.BooleanVar(value=True)
        self.show_video = tk.BooleanVar(value=False)
        self.save_check = tk.Checkbutton(master, text="Save Annotated Video", variable=self.save_video, font=("Helvetica", 10), bg="#f0f0f0")
        self.save_check.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.show_check = tk.Checkbutton(master, text="Show Video", variable=self.show_video, font=("Helvetica", 10), bg="#f0f0f0")
        self.show_check.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # Row 6: Sample Rate Selection
        tk.Label(master, text="Sample Rate (every n-th frame):", bg="#f0f0f0", font=("Helvetica", 10)).grid(row=6, column=0, padx=5, pady=5, sticky="e")
        self.sample_rate = tk.IntVar(value=32)
        self.sample_spin = tk.Spinbox(master, from_=1, to=60, textvariable=self.sample_rate, width=5, font=("Helvetica", 10))
        self.sample_spin.grid(row=6, column=1, padx=5, pady=5, sticky="w")

        # Row 7: Progress Bar
        self.progress = ttk.Progressbar(master, orient="horizontal", mode="indeterminate", length=300)
        self.progress.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

        # Row 8: Start Processing Button and Status Label
        self.start_button = tk.Button(master, text="Start Processing", command=self.start_processing, font=("Helvetica", 10, "bold"))
        self.start_button.grid(row=8, column=0, columnspan=2, pady=10)
        self.status_label = tk.Label(master, text="", bg="#f0f0f0", font=("Helvetica", 10, "italic"))
        self.status_label.grid(row=8, column=2, padx=5, pady=10)

        # Row 9: Post-processing Options
        self.play_button = tk.Button(master, text="Play Annotated Video", command=self.play_video, state="disabled", font=("Helvetica", 10))
        self.play_button.grid(row=9, column=0, padx=5, pady=5)
        self.open_log_button = tk.Button(master, text="Open Log File", command=self.open_log, state="disabled", font=("Helvetica", 10))
        self.open_log_button.grid(row=9, column=1, padx=5, pady=5)
        self.open_report_button = tk.Button(master, text="Open Report", command=self.open_report, state="disabled", font=("Helvetica", 10))
        self.open_report_button.grid(row=9, column=2, padx=5, pady=5)
        self.summarize_button = tk.Button(master, text="Summarize Report", command=self.summarize_report, state="disabled", font=("Helvetica", 10, "bold"))
        self.summarize_button.grid(row=10, column=0, padx=5, pady=5)

        # Row 10: Help Button
        self.help_button = tk.Button(master, text="Help", command=self.show_help, font=("Helvetica", 10, "bold"), bg="#d9d9d9")
        self.help_button.grid(row=10, column=2, padx=5, pady=5, sticky="e")

        self.toggle_language_options()
        self.annotated_video_path = None

    def load_video(self):
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if filepath:
            self.video_path.set(filepath)

    def toggle_language_options(self):
        state = "disabled" if self.auto_lang.get() else "readonly"
        self.primary_menu.config(state=state)
        self.secondary_menu.config(state=state)

    def start_processing(self):
        video_file = self.video_path.get()
        if not video_file or not os.path.exists(video_file):
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        # Use get_lang_code to extract the proper language code (e.g., "eng")
        if self.auto_lang.get():
            transcribe_language = None
            ocr_languages = self.get_lang_code(self.primary_lang.get())
        else:
            transcribe_language = self.get_lang_code(self.primary_lang.get())
            prim = self.get_lang_code(self.primary_lang.get())
            sec = self.get_lang_code(self.secondary_lang.get())
            ocr_languages = prim if sec == "none" else f"{prim}+{sec}"

        sample_rate = self.sample_rate.get()
        save_video = self.save_video.get()
        show_video = self.show_video.get()

        global selected_summarization_model
        selected_summarization_model = self.selected_model  # Use the dropdown selection

        self.status_label.config(text="Processing started...")
        self.start_button.config(state="disabled")
        self.progress.start()

        def processing_task():
            process_video(video_file, sample_rate=sample_rate, draw_boxes=True, save_video=save_video,
                          show_video=show_video, ocr_languages=ocr_languages)
            if save_video:
                self.annotated_video_path = os.path.splitext(video_file)[0] + "_annotated.mp4"
            self.master.after(0, self.processing_complete)

        threading.Thread(target=processing_task, daemon=True).start()

    def processing_complete(self):
        self.progress.stop()
        self.status_label.config(text="Processing completed.")
        self.start_button.config(state="normal")
        self.play_button.config(state="normal")
        self.open_log_button.config(state="normal")
        self.open_report_button.config(state="normal")
        self.summarize_button.config(state="normal")

    def get_lang_code(self, lang_display):
        if "none" in lang_display.lower():
            return "none"
        if "(" in lang_display and ")" in lang_display:
            return lang_display.split("(")[-1].split(")")[0].lower()
        return lang_display.lower()

    def play_video(self):
        if self.annotated_video_path and os.path.exists(self.annotated_video_path):
            try:
                if os.name == "nt":
                    os.startfile(self.annotated_video_path)
                else:
                    subprocess.Popen(["open", self.annotated_video_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open video: {e}")
        else:
            messagebox.showerror("Error", "Annotated video not found.")

    def open_log(self):
        log_path = os.path.abspath(LOG_FILE)
        try:
            if os.name == "nt":
                os.startfile(log_path)
            else:
                subprocess.Popen(["open", log_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open log file: {e}")

    def open_report(self):
        report_path = os.path.abspath("report.txt")
        if os.path.exists(report_path):
            try:
                if os.name == "nt":
                    os.startfile(report_path)
                else:
                    subprocess.Popen(["open", report_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open report file: {e}")
        else:
            messagebox.showerror("Error", "Report file not found.")

    def summarize_report(self):
        desc_file = generate_video_description()
        if desc_file and os.path.exists(desc_file):
            try:
                if os.name == "nt":
                    os.startfile(desc_file)
                else:
                    subprocess.Popen(["open", desc_file])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open description file: {e}")
        else:
            messagebox.showerror("Error", "Video description file not found.")

    def show_help(self):
        help_text = (
            "Video Processing GUI Help:\n\n"
            "This application processes a video file by performing the following steps:\n"
            "1. Extracts and transcribes audio using Whisper (multilingual).\n"
            "2. Detects visual objects using YOLO and generates captions using BLIP.\n"
            "3. Generates an overall narrative summary using a selected LLM (via Ollama).\n\n"
            "Usage:\n"
            "- Click 'Load Video' to select an MP4 file.\n"
            "- Choose language options for transcription.\n"
            "- Select your preferred summarization model from the dynamic list.\n"
            "- Set whether to save and/or display the annotated video and adjust the sample rate.\n"
            "- Click 'Start Processing' to begin. Monitor progress in the status label.\n"
            "- Once processing is complete, use the post-processing buttons to view the video, log, report, or summary.\n"
        )
        messagebox.showinfo("Help", help_text)


if __name__ == "__main__":
    root = tk.Tk()
    gui = VideoProcessingGUI(root)
    root.mainloop()
