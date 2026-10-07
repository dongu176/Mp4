import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict
import numpy as np


class CustomJSONEncoder(json.JSONEncoder):
    """NumPy 타입 및 dataclass를 JSON으로 직렬화하기 위한 인코더."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class NoteEvent:
    type: str = "note"  # note, rest, chord
    position: str = "0/1"  # "0/1", "1/4", "1/2", "3/4" 등
    duration: str = "1/4"
    string: Optional[int] = None  # 1 ~ 6
    fret: Optional[int] = None  # 0 ~ 24 or None if unknown
    technique: Optional[str] = None  # hammer-on, pull-off, slide, bend 등
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SourceInfo:
    timestamp_start: float
    timestamp_end: float
    frame_index: int
    segment_id: int
    image_path: str
    bbox: List[int]  # [ymin, xmin, ymax, xmax]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MeasureData:
    number: int
    source: SourceInfo
    time_signature: Dict[str, int] = field(default_factory=lambda: {"numerator": 4, "denominator": 4})
    confidence: float = 1.0
    recognized_measure_number: Optional[int] = None
    events: List[NoteEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "number": self.number,
            "source": self.source.to_dict(),
            "time_signature": self.time_signature,
            "confidence": self.confidence,
            "recognized_measure_number": self.recognized_measure_number,
            "events": [e.to_dict() if isinstance(e, NoteEvent) else e for e in self.events]
        }
        return res


@dataclass
class ScoreMetadata:
    title: Optional[str] = None
    artist: Optional[str] = None
    source_url: Optional[str] = None
    expected_measures: int = 112
    detected_measures: int = 0


@dataclass
class ScoreData:
    schema_version: str = "1.0"
    metadata: ScoreMetadata = field(default_factory=ScoreMetadata)
    instrument: Dict[str, Any] = field(default_factory=lambda: {
        "type": "guitar",
        "strings": 6,
        "tuning": ["E", "A", "D", "G", "B", "E"]
    })
    score: Dict[str, Any] = field(default_factory=lambda: {
        "tempo": 120,
        "time_signature": {"numerator": 4, "denominator": 4},
        "measures": []
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": asdict(self.metadata),
            "instrument": self.instrument,
            "score": {
                "tempo": self.score.get("tempo"),
                "time_signature": self.score.get("time_signature"),
                "measures": [
                    m.to_dict() if isinstance(m, MeasureData) else m
                    for m in self.score.get("measures", [])
                ]
            }
        }
