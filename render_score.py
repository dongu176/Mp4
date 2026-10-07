import sys
import os
import json
from models import CustomJSONEncoder
from validation import validate_score_data
from renderer import SVGTabRenderer


def render_from_json(json_path: str):
    """score.json을 입력받아 독립적으로 검증 및 SVG/PDF 재렌더링을 수행합니다."""
    if not os.path.exists(json_path):
        print(f"❌ 오류: 파일이 존재하지 않습니다: {json_path}")
        sys.exit(1)

    print(f"📄 Score JSON 로드 중: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        score_dict = json.load(f)

    # 1. 검증 수행
    print("🔍 Score Validation 실행 중...")
    val_result = validate_score_data(score_dict)
    val_output_path = "output/score_validation.json"
    with open(val_output_path, "w", encoding="utf-8") as f:
        json.dump(val_result, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

    summary = val_result["summary"]
    print(f"   - Expected measures : {summary.get('expected_measures')}")
    print(f"   - Detected measures : {summary.get('detected_measures')}")
    print(f"   - Missing measures  : {summary.get('missing_measures')}")

    if val_result["warnings"]:
        print(f"⚠️ Validation 경고 {len(val_result['warnings'])}개 발견 (score_validation.json 참조)")

    # 2. SVG 렌더링
    print("🎨 Vector SVG 악보 생성 중...")
    renderer = SVGTabRenderer(score_dict, measures_per_line=4)
    svg_path = renderer.render_svg("output/score.svg")
    print(f"✅ SVG 생성 완료: {svg_path}")

    # 3. PDF 렌더링
    print("🖨️ PDF 악보 변환 중...")
    pdf_path = "output/score.pdf"
    ok = renderer.render_pdf(svg_path, pdf_path)
    if ok:
        print(f"🎉 PDF 생성 완료: {pdf_path}")
    else:
        print("❌ PDF 변환에 실패했습니다. (CairoSVG 패키지 및 C 라이브러리를 확인하세요)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python render_score.py output/score.json")
        sys.exit(1)
    render_from_json(sys.argv[1])
