import json
import os
from typing import Dict, Any, List


SCORE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Guitar TAB Score Schema",
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "metadata": {
            "type": "object",
            "properties": {
                "title": {"type": ["string", "null"]},
                "artist": {"type": ["string", "null"]},
                "source_url": {"type": ["string", "null"]},
                "expected_measures": {"type": "integer"},
                "detected_measures": {"type": "integer"}
            },
            "required": ["expected_measures", "detected_measures"]
        },
        "instrument": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "strings": {"type": "integer"},
                "tuning": {"type": "array", "items": {"type": "string"}}
            }
        },
        "score": {
            "type": "object",
            "properties": {
                "tempo": {"type": ["number", "null"]},
                "time_signature": {"type": "object"},
                "measures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "number": {"type": "integer"},
                            "source": {"type": "object"},
                            "confidence": {"type": "number"},
                            "events": {"type": "array"}
                        },
                        "required": ["number", "source", "events"]
                    }
                }
            },
            "required": ["measures"]
        }
    },
    "required": ["schema_version", "metadata", "score"]
}


def write_schema_file(output_path: str = "output/score.schema.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(SCORE_SCHEMA, f, indent=2, ensure_ascii=False)


def validate_score_data(score_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Score JSON 데이터를 검증하고 경고 및 수치 요약을 반환합니다."""
    write_schema_file()
    
    validation_result = {
        "is_valid_schema": True,
        "warnings": [],
        "missing_regions": [],
        "summary": {}
    }

    try:
        import jsonschema
        jsonschema.validate(instance=score_dict, schema=SCORE_SCHEMA)
    except Exception as e:
        validation_result["is_valid_schema"] = False
        validation_result["warnings"].append(f"JSON Schema Validation Error: {str(e)}")

    metadata = score_dict.get("metadata", {})
    expected = metadata.get("expected_measures", 112)
    measures = score_dict.get("score", {}).get("measures", [])
    detected = len(measures)

    validation_result["summary"] = {
        "expected_measures": expected,
        "detected_measures": detected,
        "missing_measures": max(0, expected - detected)
    }

    # 1. 마디 번호 연속성 검사
    numbers = [m.get("number") for m in measures]
    for i in range(1, len(numbers)):
        if numbers[i] != numbers[i - 1] + 1:
            validation_result["warnings"].append(
                f"Discontinuity in measure sequence: {numbers[i-1]} -> {numbers[i]}"
            )

    # 2. 타임스탬프 역전 검사 및 누락 구간 추정
    last_ts = 0.0
    for m in measures:
        src = m.get("source", {})
        ts_start = src.get("timestamp_start", 0.0)
        ts_end = src.get("timestamp_end", 0.0)

        if ts_start < last_ts:
            validation_result["warnings"].append(
                f"Timestamp reversal at measure {m.get('number')}: {ts_start}s < {last_ts}s"
            )
        
        # 8초 이상 악보 캡처 공백이 발생하는 경우 누락 가능 구간으로 기록
        if ts_start - last_ts > 8.0 and last_ts > 0:
            validation_result["missing_regions"].append({
                "from_timestamp": round(last_ts, 2),
                "to_timestamp": round(ts_start, 2),
                "segment_id": src.get("segment_id", 0)
            })
        last_ts = ts_end

    # 3. Note 유효성 검사
    for m in measures:
        for ev in m.get("events", []):
            st = ev.get("string")
            fr = ev.get("fret")
            conf = ev.get("confidence", 1.0)
            
            if st is not None and not (1 <= st <= 6):
                validation_result["warnings"].append(
                    f"Measure {m.get('number')}: Invalid string number {st}"
                )
            if fr is not None and not (0 <= fr <= 24):
                validation_result["warnings"].append(
                    f"Measure {m.get('number')}: Invalid fret number {fr}"
                )
            if conf < 0.0 or conf > 1.0:
                validation_result["warnings"].append(
                    f"Measure {m.get('number')}: Confidence out of bounds {conf}"
                )

    return validation_result
