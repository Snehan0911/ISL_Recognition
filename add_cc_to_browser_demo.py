"""
add_cc_to_browser_demo.py: Adds professional, burned-in Closed Captions (CC) and explanations
to the IndiSign AI Browser Live Demo video, and exports an .srt subtitle file.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

INPUT_VIDEO = "IndiSign_AI_Browser_Live_Demo.mp4"
OUTPUT_VIDEO = "IndiSign_AI_Browser_Live_Demo_CC.mp4"
OUTPUT_SRT = "IndiSign_AI_Browser_Live_Demo.srt"

WIDTH = 1920
HEIGHT = 1080

# --- Caption Timeline (start_sec, end_sec, caption_text, highlight_label) ---
CAPTIONS = [
    (0.0, 4.5, "Loading IndiSign AI — Real-Time Indian Sign Language Classifier in Chrome", "LIVE BROWSER"),
    (4.5, 9.0, "Real-Time AI Inference: Recognizing Sign 'A' with 98.26% confidence in 19.8 ms", "AI INFERENCE"),
    (9.0, 13.0, "Appending detected letter 'A' into the assistive Word & Sentence Builder", "SENTENCE BUILDER"),
    (13.0, 17.5, "Random Sign Testing: Rapid classification & continuous sequence composition", "EDGE INFERENCE"),
    (17.5, 21.0, "Speech Synthesis: Web Speech API vocalizes formulated sign sentences for hearing audiences", "TEXT-TO-SPEECH"),
    (21.0, 26.5, "Visual ISL Gallery: Verified real photo dictionary for all 24 static signs (A to Y)", "ISL DICTIONARY"),
    (26.5, 30.0, "Architectural Note: Dynamic signs 'J' & 'Z' require 3D motion tracing and are omitted by design", "DESIGN NOTE"),
    (30.0, 35.5, "Quantitative Evaluation: 100.00% test accuracy & 0.0018 loss across 482 unseen test images", "BENCHMARK"),
    (35.5, 41.0, "Cloud Deployment: Live on Streamlit Cloud & backed by high-throughput FastAPI REST engine", "DEPLOYMENT")
]

# --- Font setup ---
FONT_PATH = "C:/Windows/Fonts/segoeuib.ttf"
FONT_REGULAR = "C:/Windows/Fonts/segoeui.ttf"

font_cc_label = ImageFont.truetype(FONT_PATH, 18)
font_cc_text = ImageFont.truetype(FONT_PATH, 24)


def generate_srt_file():
    """Generates a standard SubRip (.srt) subtitle file."""
    def format_time(seconds):
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    lines = []
    for idx, (start, end, text, label) in enumerate(CAPTIONS, 1):
        lines.append(f"{idx}")
        lines.append(f"{format_time(start)} --> {format_time(end)}")
        lines.append(f"[{label}] {text}\n")

    with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SUCCESS] Subtitle file generated: '{OUTPUT_SRT}'")


def get_current_caption(time_sec):
    for start, end, text, label in CAPTIONS:
        if start <= time_sec < end:
            return text, label
    return None, None


def draw_caption_overlay(pil_img, text, label):
    """Draws a sleek, modern, high-contrast closed caption banner at the bottom."""
    draw = ImageDraw.Draw(pil_img)

    # Measure text
    label_str = f"CC  |  {label}"
    label_bbox = font_cc_label.getbbox(label_str)
    label_w = label_bbox[2] - label_bbox[0]

    text_bbox = font_cc_text.getbbox(text)
    text_w = text_bbox[2] - text_bbox[0]

    content_w = max(label_w + 30, text_w) + 70
    banner_w = min(content_w, WIDTH - 120)
    banner_h = 80
    bx = (WIDTH - banner_w) // 2
    by = HEIGHT - 110  # Positioned 30px above bottom edge

    # Background card with glow
    draw.rounded_rectangle(
        [bx - 2, by - 2, bx + banner_w + 2, by + banner_h + 2],
        radius=20,
        fill=(56, 189, 248, 60)
    )
    draw.rounded_rectangle(
        [bx, by, bx + banner_w, by + banner_h],
        radius=18,
        fill=(11, 15, 25, 240),
        outline=(56, 189, 248),
        width=2
    )

    # Badge icon
    draw.rounded_rectangle([bx + 20, by + 12, bx + 20 + label_w + 20, by + 34], radius=6, fill=(245, 158, 11))
    draw.text((bx + 30, by + 14), label_str, font=font_cc_label, fill=(0, 0, 0))

    # Caption text
    draw.text((bx + 20, by + 42), text, font=font_cc_text, fill=(241, 245, 249))


def process_video():
    generate_srt_file()

    print(f"[INFO] Opening input video: '{INPUT_VIDEO}'...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open '{INPUT_VIDEO}'. Please ensure the file exists.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video loaded: {WIDTH}x{HEIGHT} @ {fps} FPS, {total_frames} frames (~{total_frames/fps:.1f}s)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (WIDTH, HEIGHT))

    frame_num = 0
    print("[INFO] Burning Closed Captions (CC) into video frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_num / fps
        caption_text, caption_label = get_current_caption(time_sec)

        if caption_text:
            # Convert OpenCV BGR to PIL RGB
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw_caption_overlay(pil_img, caption_text, caption_label)
            # Convert back to BGR
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()

    size_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
    print("=" * 65)
    print(f"[SUCCESS] CC Video Successfully Created: '{OUTPUT_VIDEO}'")
    print(f"          Resolution  : {WIDTH}x{HEIGHT}")
    print(f"          Total Frames: {frame_num}")
    print(f"          File Size   : {size_mb:.2f} MB")
    print(f"          Subtitles   : '{OUTPUT_SRT}'")
    print("=" * 65)


if __name__ == "__main__":
    process_video()

