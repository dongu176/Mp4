import sys
import os
import glob
import json
import time
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import yt_dlp
from google import genai
from google.genai import types

from models import ScoreData, CustomJSONEncoder
from deduplication import process_deduplication_and_sequence
from validation import validate_score_data
from renderer import SVGTabRenderer

# ------------------------------------------------------------------
# Global Config
# ------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_env_model = os.environ.get("GEMINI_MODEL", "").strip()
MODEL_CANDIDATES = ([_env_model] if _env_model else []) + [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
]
_model_idx = 0
SAMPLE_SEC = float(os.environ.get("SAMPLE_SEC", "0.10"))  # 안전한 overlap 확보를 위해 0.1초 적용
VIDEO_HEIGHT = int(os.environ.get("VIDEO_HEIGHT", "1080"))
EXPECTED_MEASURES = int(os.environ.get("EXPECTED_MEASURES", "112"))

if not GEMINI_API_KEY:
    print("❌ 오류: GEMINI_API_KEY 환경변수가 설정되지 않았증니다.")
    sys.exit(1)

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=60_000),
)

PROMPT_BOX = """
Identify the guitar TAB (tablature) score area in this video frame.
Return ONLY a JSON object with normalized coordinates (0 to 1000 scale):
{"has_tab": true, "box_2d": [ymin, xmin, ymax, xmax]}
If no TAB score is present, return {"has_tab": false, "box_2d": []}.
"""


# ------------------------------------------------------------------
# 1. Video Download & Processing
# ------------------------------------------------------------------
import os
import argparse
import yt_dlp

def download_youtube_video(url: str, output_path: str = "input_video.mp4") -> str:
    """
    YouTube 영상을 cookies.txt 기반으로 안전하게 다운로드합니다.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'cookiefile': 'cookies.txt',  # 비밀 변수로 생성된 쿠키 파일 적용
        'quiet': False,
        'no_warnings': True,
    }

    print(f"[*] Downloading video from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    print(f"[+] Download completed: {output_path}")
    return output_path


def open_video(path: str) -> Tuple[cv2.VideoCapture, str]:
    cap = cv2.VideoCapture(path)
    ok, _ = cap.read() if cap.isOpened() else (False, None)
    if ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap, path

    print("⚠️ OpenCV 코덱 호환성 문제로 ffmpeg 변환을 진행합니다...")
    cap.release()
    converted = "video_h264.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", converted],
        check=True,
    )
    cap = cv2.VideoCapture(converted)
    if not cap.isOpened():
        raise RuntimeError("영상 파일을 열 수 없습니다.")
    return cap, converted


# ------------------------------------------------------------------
# 2. Segment-based TAB Detection
# ------------------------------------------------------------------
def detect_tab_box_segment(frame_bgr: np.ndarray) -> Optional[List[float]]:
    """CV 오선 헤비스틱 검출 후 실패 시 Gemini로 보완하는 구간별 영역 탐지."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=int(w * 0.3), maxLineGap=10)

    if lines is not None and len(lines) >= 4:
        ys = [l[0][1] for l in lines] + [l[0][3] for l in lines]
        ymin, ymax = min(ys), max(ys)
        if ymax - ymin > 30:
            pad = int((ymax - ymin) * 0.2)
            return [
                max(0.0, (ymin - pad) / h * 1000),
                0.0,
                min(1000.0, (ymax + pad) / h * 1000),
                1000.0
            ]

    # Gemini Fallback
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    pil_img.thumbnail((1280, 1280))
    global _model_idx
    model = MODEL_CANDIDATES[_model_idx % len(MODEL_CANDIDATES)]
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[pil_img, PROMPT_BOX],
            config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
        )
        data = json.loads(resp.text)
        if isinstance(data, list):
            data = data[0] if data else {}
        box = data.get("box_2d", [])
        if data.get("has_tab") and len(box) == 4:
            return [float(v) for v in box]
    except Exception:
        pass
    return None


# ------------------------------------------------------------------
# 3. Measure Extraction & Frame Processing
# ------------------------------------------------------------------
def extract_raw_measures_from_frame(
    frame_bgr: np.ndarray,
    box: List[float],
    timestamp: float,
    frame_idx: int,
    segment_id: int,
    output_dir: str
) -> List[Dict[str, Any]]:
    h, w = frame_bgr.shape[:2]
    ymin, xmin, ymax, xmax = box
    top = max(0, int(ymin / 1000 * h))
    bottom = min(h, int(ymax / 1000 * h))
    left = max(0, int(xmin / 1000 * w))
    right = min(w, int(xmax / 1000 * w))

    crop = frame_bgr[top:bottom, left:right]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 50:
        return []

    # 수직 마디선 탐지
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    # Vertical bar line kernel
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(crop.shape[0] * 0.6)))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    col_sums = np.sum(v_lines > 0, axis=0)
    bar_xs = np.where(col_sums > (crop.shape[0] * 0.5))[0]

    # 그룹화
    merged_bars = []
    for x in bar_xs:
        if not merged_bars or x - merged_bars[-1] > 15:
            merged_bars.append(x)

    all_xs = [0] + merged_bars + [crop.shape[1]]
    extracted = []

    for i in range(len(all_xs) - 1):
        x1, x2 = all_xs[i], all_xs[i + 1]
        width = x2 - x1
        if width < crop.shape[1] * 0.15:  # 지나치게 좁은 조각 제외
            continue

        m_crop = crop[:, x1:x2]
        raw_id = f"raw_{frame_idx}_{i}"
        img_filename = f"{raw_id}.png"
        img_path = os.path.join(output_dir, "raw_measures", img_filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, m_crop)

        extracted.append({
            "raw_id": raw_id,
            "timestamp_start": timestamp,
            "timestamp_end": timestamp + SAMPLE_SEC,
            "frame_index": frame_idx,
            "segment_id": segment_id,
            "image_path": img_path,
            "bbox": [top, left + x1, bottom, left + x2],
            "confidence": 0.90,
            "notes": []  # Raw Recognition 단계에서는 추후 구조 인식 정보 삽입 가능
        })

    return extracted


# ------------------------------------------------------------------
# Main Orchestration Pipeline
# ------------------------------------------------------------------
def run_pipeline(url: str):
    os.makedirs("output/raw_measures", exist_ok=True)
    os.makedirs("output/debug", exist_ok=True)

    print("[1/8] Video download")
    video_path, video_title = download_video(url)
    cap, _ = open_video(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"      Duration: {duration:.1f}s, FPS: {fps:.1f}, Frames: {total_frames}")

    print("[2/8] TAB region detection (Segmented)")
    # 세그먼트 생성 (영상 길이에 따라 4~6개 세그먼트로 분할)
    num_segments = 4
    seg_frames = total_frames // num_segments
    segment_boxes: Dict[int, List[float]] = {}

    for seg_id in range(num_segments):
        target_frame = int((seg_id + 0.5) * seg_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if ok:
            box = detect_tab_box_segment(frame)
            if box:
                segment_boxes[seg_id] = box
            else:
                segment_boxes[seg_id] = [200.0, 0.0, 800.0, 1000.0]  # 기본값

    print(f"      Detected BBoxes across {len(segment_boxes)} segments")

    print("[3/8] Frame extraction & [4/8] Measure detection")
    step_frames = max(1, int(fps * SAMPLE_SEC))
    raw_measures_list: List[Dict[str, Any]] = []

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step_frames == 0:
            timestamp = frame_idx / fps
            current_seg = min(num_segments - 1, int(frame_idx // seg_frames))
            box = segment_boxes.get(current_seg, [200.0, 0.0, 800.0, 1000.0])

            measures = extract_raw_measures_from_frame(
                frame, box, timestamp, frame_idx, current_seg, "output"
            )
            raw_measures_list.extend(measures)

        frame_idx += 1

    cap.release()

    print("[5/8] TAB recognition -> Writing output/raw_recognition.json")
    raw_recognition_path = "output/raw_recognition.json"
    with open(raw_recognition_path, "w", encoding="utf-8") as f:
        json.dump(raw_measures_list, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

    print("[6/8] Deduplication / sequence restoration")
    score_data_obj = process_deduplication_and_sequence(raw_measures_list, EXPECTED_MEASURES)
    score_data_obj.metadata.title = video_title
    score_dict = score_data_obj.to_dict()

    score_json_path = "output/score.json"
    with open(score_json_path, "w", encoding="utf-8") as f:
        json.dump(score_dict, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

    print("[7/8] Score validation")
    val_result = validate_score_data(score_dict)
    with open("output/score_validation.json", "w", encoding="utf-8") as f:
        json.dump(val_result, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

    print("[8/8] SVG / PDF rendering")
    renderer = SVGTabRenderer(score_dict, measures_per_line=4)
    svg_path = renderer.render_svg("output/score.svg")
    renderer.render_pdf(svg_path, "output/score.pdf")

    # Final Execution Metrics Print
    detected = score_dict["metadata"]["detected_measures"]
    missing = max(0, EXPECTED_MEASURES - detected)
    print("\n=========================================")
    print(f"Expected measures : {EXPECTED_MEASURES}")
    print(f"Raw measures      : {len(raw_measures_list)}")
    print(f"After dedup       : {detected}")
    print(f"Final measures    : {detected}")
    print(f"Missing measures  : {missing}")
    print("=========================================\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python extract.py \"YOUTUBE_URL\"")
        sys.exit(1)
    run_pipeline(sys.argv[1])
