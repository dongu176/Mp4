import os
import cv2
import numpy as np
from typing import List, Dict, Any
from models import MeasureData, SourceInfo, NoteEvent, ScoreData, ScoreMetadata


def compute_perceptual_hash(image_path: str) -> str:
    """이미지의 Perceptual Hash(Difference Hash)를 계산합니다."""
    if not os.path.exists(image_path):
        return "0000000000000000"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return "0000000000000000"
    resized = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return "".join(["1" if b else "0" for b in diff.flatten()])


def hamming_distance(hash1: str, hash2: str) -> int:
    """두 해시 값 간의 해밍 거리를 계산합니다."""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def process_deduplication_and_sequence(
    raw_measures: List[Dict[str, Any]],
    expected_measures: int = 112
) -> ScoreData:
    """
    Raw Recognition 데이터를 입력받아:
    1. 동일 마디의 연속적 비디오 프레임 중복 (Visual Duplicate Frame) 제거
    2. 음악적 반복 구간(Musical Repeat) 보존
    3. 순차적 마디 번호 부여 및 score.json 생성
    """
    if not raw_measures:
        score_obj = ScoreData()
        score_obj.metadata.expected_measures = expected_measures
        score_obj.metadata.detected_measures = 0
        return score_obj

    # 1. 타임스탬프 순 정렬
    sorted_raw = sorted(raw_measures, key=lambda x: (x["timestamp_start"], x["frame_index"]))

    deduped_raw: List[Dict[str, Any]] = []

    for raw in sorted_raw:
        image_path = raw.get("image_path", "")
        p_hash = compute_perceptual_hash(image_path)
        raw["p_hash"] = p_hash

        if not deduped_raw:
            deduped_raw.append(raw)
            continue

        prev = deduped_raw[-1]
        time_diff = abs(raw["timestamp_start"] - prev["timestamp_start"])
        h_dist = hamming_distance(raw["p_hash"], prev["p_hash"])

        # 시각적으로 거의 동일(해밍 거리 <= 6)하고, 타임스탬프 간격이 3초 미만이면 동일 프레임 캡처 중복으로 판단
        if time_diff < 3.0 and h_dist <= 6:
            # 신뢰도가 더 높은 항목으로 교체
            if raw.get("confidence", 0) > prev.get("confidence", 0):
                deduped_raw[-1] = raw
        else:
            # 음악적으로 동일한 리프라도 타임스탬프가 떨어져 있거나 시각적 차이가 나면 별도 마디로 보존
            deduped_raw.append(raw)

    # 2. 순차적 마디 생성 및 정규화
    final_measures: List[MeasureData] = []
    for idx, item in enumerate(deduped_raw, start=1):
        src = SourceInfo(
            timestamp_start=item["timestamp_start"],
            timestamp_end=item["timestamp_end"],
            frame_index=item["frame_index"],
            segment_id=item["segment_id"],
            image_path=item["image_path"],
            bbox=item["bbox"]
        )

        events_raw = item.get("notes", [])
        note_events = []
        for ev in events_raw:
            note_events.append(NoteEvent(
                type=ev.get("type", "note"),
                position=ev.get("position", "0/1"),
                duration=ev.get("duration", "1/4"),
                string=ev.get("string"),
                fret=ev.get("fret"),
                technique=ev.get("technique"),
                confidence=ev.get("confidence", 1.0)
            ))

        measure_obj = MeasureData(
            number=idx,
            source=src,
            time_signature=item.get("time_signature", {"numerator": 4, "denominator": 4}),
            confidence=item.get("confidence", 1.0),
            recognized_measure_number=item.get("recognized_measure_number"),
            events=note_events
        )
        final_measures.append(measure_obj)

    score_data = ScoreData()
    score_data.metadata.expected_measures = expected_measures
    score_data.metadata.detected_measures = len(final_measures)
    score_data.score["measures"] = final_measures

    return score_data
