"""
generate_browser_live_demo.py: Records a real live browser demonstration of IndiSign AI,
adds a realistic Google Chrome browser frame (address bar + tabs), and compiles an MP4 video.
"""

import os
import time
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

OUTPUT_FINAL_MP4 = "IndiSign_AI_Browser_Live_Demo.mp4"
TEMP_REC_DIR = "temp_browser_recording"

# Dimensions
VIDEO_W = 1920
VIDEO_H = 1080
HEADER_H = 80          # Chrome address bar + tabs height
BROWSER_H = VIDEO_H - HEADER_H  # 1000px

# Font
FONT_DIR = "C:/Windows/Fonts"
def get_font(name="segoeui.ttf", size=16):
    path = os.path.join(FONT_DIR, name)
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()

font_tab = get_font("segoeui.ttf", 14)
font_url = get_font("segoeui.ttf", 15)
font_icon = get_font("segoeuib.ttf", 16)


def create_chrome_header():
    """Renders a sleek Google Chrome dark-mode header bar (80px tall)."""
    header = Image.new("RGB", (VIDEO_W, HEADER_H), color=(32, 33, 36))
    draw = ImageDraw.Draw(header)

    # Top Tab Bar (0 to 38px)
    # Window control buttons (top right)
    draw.text((VIDEO_W - 110, 10), "—", font=font_icon, fill=(154, 160, 166))
    draw.text((VIDEO_W - 70, 10), "□", font=font_icon, fill=(154, 160, 166))
    draw.text((VIDEO_W - 30, 10), "✕", font=font_icon, fill=(154, 160, 166))

    # Active Tab (IndiSign AI)
    tab_x, tab_w = 80, 260
    draw.rounded_rectangle([tab_x, 8, tab_x + tab_w, 40], radius=8, fill=(44, 47, 51))
    draw.text((tab_x + 16, 14), "🤟 IndiSign AI | Indian Sign L...", font=font_tab, fill=(232, 234, 237))
    draw.text((tab_x + tab_w - 24, 14), "✕", font=font_tab, fill=(154, 160, 166))

    # Bottom Address Bar Area (38 to 80px)
    draw.rectangle([0, 38, VIDEO_W, HEADER_H], fill=(44, 47, 51))

    # Navigation arrows: Back, Forward, Reload
    draw.text((25, 48), "←", font=font_icon, fill=(154, 160, 166))
    draw.text((65, 48), "→", font=font_icon, fill=(112, 117, 122))
    draw.text((105, 48), "⟳", font=font_icon, fill=(154, 160, 166))

    # URL Bar
    url_x = 150
    url_w = VIDEO_W - 250
    draw.rounded_rectangle([url_x, 44, url_x + url_w, 74], radius=15, fill=(32, 33, 36))
    
    # SSL Lock + URL text
    draw.text((url_x + 16, 50), "🔒", font=font_tab, fill=(52, 211, 153))
    draw.text((url_x + 44, 50), "https://islrecognition-ezwzp8lmwbrh2cjbaratev.streamlit.app", font=font_url, fill=(232, 234, 237))

    # Profile / extension icons on far right
    draw.text((VIDEO_W - 75, 48), "🧩", font=font_tab, fill=(154, 160, 166))
    draw.ellipse([VIDEO_W - 40, 47, VIDEO_W - 16, 71], fill=(99, 102, 241))
    draw.text((VIDEO_W - 33, 49), "S", font=font_icon, fill=(255, 255, 255))

    return header

CHROME_HEADER = create_chrome_header()


def record_live_browser():
    os.makedirs(TEMP_REC_DIR, exist_ok=True)

    print("[INFO] Launching Google Chrome via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True
        )

        context = browser.new_context(
            viewport={"width": VIDEO_W, "height": BROWSER_H},
            record_video_dir=TEMP_REC_DIR,
            record_video_size={"width": VIDEO_W, "height": BROWSER_H}
        )

        page = context.new_page()

        print("[INFO] Navigating to Streamlit live application (http://localhost:8501)...")
        page.goto("http://localhost:8501")
        page.wait_for_selector("text=Recognized Sign", timeout=40000)
        print("[INFO] App loaded and initialized!")
        page.wait_for_timeout(2500)

        # ---------------------------------------------
        # STEP 1: Hero Banner & Benchmark Sidebar View (3s)
        # ---------------------------------------------
        print("[INFO] Simulating user interaction: Inspecting Hero & Sidebar...")
        page.wait_for_timeout(2000)

        # ---------------------------------------------
        # STEP 2: Test Sign 'A' and Append to Sentence
        # ---------------------------------------------
        print("[INFO] Interacting: Appending Sign 'A'...")
        append_btn = page.locator("button:has-text('Append Letter')").first
        if append_btn.count() > 0:
            append_btn.click()
            page.wait_for_timeout(2000)

        # ---------------------------------------------
        # STEP 3: Click 'Random' to demonstrate instant inference on another sign
        # ---------------------------------------------
        print("[INFO] Clicking '🎲 Random' button for instant dynamic inference...")
        random_btn = page.locator("button:has-text('Random')").first
        if random_btn.count() > 0:
            random_btn.click()
            page.wait_for_timeout(2500)
            
            # Append the newly recognized random sign!
            append_btn = page.locator("button:has-text('Append Letter')").first
            if append_btn.count() > 0:
                append_btn.click()
                page.wait_for_timeout(2000)

        # Click Random once more to showcase speed
        if random_btn.count() > 0:
            random_btn.click()
            page.wait_for_timeout(2500)
            append_btn = page.locator("button:has-text('Append Letter')").first
            if append_btn.count() > 0:
                append_btn.click()
                page.wait_for_timeout(2000)

        # ---------------------------------------------
        # STEP 4: Sentence Builder Controls & Text-to-Speech
        # ---------------------------------------------
        print("[INFO] Testing Sentence Builder: Adding space and clicking 'Speak Aloud'...")
        space_btn = page.locator("button:has-text('Add Space')").first
        if space_btn.count() > 0:
            space_btn.click()
            page.wait_for_timeout(1000)

        speak_btn = page.locator("button:has-text('Speak Aloud')").first
        if speak_btn.count() > 0:
            speak_btn.click()
            page.wait_for_timeout(3000)  # Wait to show speaking toast

        # ---------------------------------------------
        # STEP 5: Switch to Tab 2: Visual ISL Gallery
        # ---------------------------------------------
        print("[INFO] Navigating to Tab 2: Sign Alphabet Reference Gallery...")
        tab_buttons = page.locator("[role='tab']")
        if tab_buttons.count() >= 2:
            tab_buttons.nth(1).click()
            page.wait_for_timeout(2000)

            # Smoothly scroll down through the 24 alphabet cards
            print("[INFO] Scrolling through visual alphabet flashcards...")
            for scroll_y in [300, 700, 1200]:
                page.evaluate(f"window.scrollTo({{top: {scroll_y}, behavior: 'smooth'}})")
                page.wait_for_timeout(1500)

            page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
            page.wait_for_timeout(1500)

        # ---------------------------------------------
        # STEP 6: Switch to Tab 3: Performance & Evaluation Analytics
        # ---------------------------------------------
        print("[INFO] Navigating to Tab 3: Model Evaluation Analytics...")
        if tab_buttons.count() >= 3:
            tab_buttons.nth(2).click()
            page.wait_for_timeout(2000)

            # Scroll down to training curves & confusion matrix
            page.evaluate("window.scrollTo({top: 400, behavior: 'smooth'})")
            page.wait_for_timeout(3000)
            page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
            page.wait_for_timeout(1500)

        # ---------------------------------------------
        # STEP 7: Switch to Tab 4: Cloud Deployment & API
        # ---------------------------------------------
        print("[INFO] Navigating to Tab 4: Cloud Deployment & API...")
        if tab_buttons.count() >= 4:
            tab_buttons.nth(3).click()
            page.wait_for_timeout(3000)

        # Final pause on home tab
        tab_buttons.nth(0).click()
        page.wait_for_timeout(2500)

        # Save and close
        print("[INFO] Finalizing browser recording...")
        page.close()
        raw_video_path = page.video.path()
        context.close()
        browser.close()

    print(f"[SUCCESS] Raw browser video saved to: {raw_video_path}")
    return raw_video_path


def compose_final_video(raw_webm_path):
    """
    Reads the raw browser WebM recording, attaches the realistic Google Chrome header bar,
    and encodes the result into a universal 1080p MP4.
    """
    print(f"[INFO] Composing final 1080p video with Google Chrome address bar: '{OUTPUT_FINAL_MP4}'...")
    cap = cv2.VideoCapture(raw_webm_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_FINAL_MP4, fourcc, fps, (VIDEO_W, VIDEO_H))

    header_np = np.array(CHROME_HEADER)  # RGB
    header_bgr = cv2.cvtColor(header_np, cv2.COLOR_RGB2BGR)

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize browser frame to exact width and browser area height
        browser_content = cv2.resize(frame, (VIDEO_W, BROWSER_H), interpolation=cv2.INTER_LINEAR)

        # Stack Chrome header on top of browser content -> exactly 1920x1080!
        full_canvas = np.vstack([header_bgr, browser_content])

        out.write(full_canvas)
        frame_idx += 1

    cap.release()
    out.release()
    elapsed = time.time() - t0
    size_mb = os.path.getsize(OUTPUT_FINAL_MP4) / (1024 * 1024)

    print("=" * 65)
    print(f"[SUCCESS] Final Live Browser Demo Video Ready: '{OUTPUT_FINAL_MP4}'")
    print(f"          Resolution  : {VIDEO_W}x{VIDEO_H} (Full HD)")
    print(f"          Total Frames: {frame_idx} (~{frame_idx/fps:.1f} seconds)")
    print(f"          File Size   : {size_mb:.2f} MB")
    print(f"          Encoding    : {elapsed:.1f} s ({frame_idx/elapsed:.1f} FPS)")
    print("=" * 65)

    # Clean up temp recordings folder
    shutil.rmtree(TEMP_REC_DIR, ignore_errors=True)


if __name__ == "__main__":
    raw_video = record_live_browser()
    compose_final_video(raw_video)

