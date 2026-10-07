import os
import xml.etree.ElementTree as ET
from typing import Dict, Any, List


class SVGTabRenderer:
    """Score JSON을 입력받아 벡터 SVG 악보 및 PDF를 생성하는 레이아웃 엔진."""

    def __init__(self, score_dict: Dict[str, Any], measures_per_line: int = 4):
        self.score_dict = score_dict
        self.measures_per_line = measures_per_line
        self.page_width = 1240
        self.margin = 70
        self.content_width = self.page_width - (self.margin * 2)
        self.staff_height = 100  # 6줄 오선 높이 (간격 20px)
        self.line_spacing = 180  # 줄간 거리

    def render_svg(self, output_svg_path: str = "output/score.svg") -> str:
        os.makedirs(os.path.dirname(output_svg_path), exist_ok=True)
        measures = self.score_dict.get("score", {}).get("measures", [])
        total_measures = len(measures)

        lines_count = (total_measures + self.measures_per_line - 1) // self.measures_per_line
        header_height = 140
        total_height = header_height + (lines_count * self.line_spacing) + 100

        svg = ET.Element("svg", {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(self.page_width),
            "height": str(total_height),
            "viewBox": f"0 0 {self.page_width} {total_height}"
        })

        # 배경 (흰색)
        ET.SubElement(svg, "rect", {
            "x": "0", "y": "0",
            "width": str(self.page_width), "height": str(total_height),
            "fill": "#FFFFFF"
        })

        # 헤더 (제목 및 메타데이터)
        title_text = self.score_dict.get("metadata", {}).get("title") or "Guitar TAB Score"
        title_elem = ET.SubElement(svg, "text", {
            "x": str(self.page_width / 2),
            "y": "60",
            "font-family": "DejaVu Sans, NanumGothic, sans-serif",
            "font-size": "32",
            "font-weight": "bold",
            "text-anchor": "middle",
            "fill": "#000000"
        })
        title_elem.text = title_text

        # 마디별 배치
        y_cursor = header_height

        for line_idx in range(lines_count):
            line_measures = measures[line_idx * self.measures_per_line : (line_idx + 1) * self.measures_per_line]
            m_count = len(line_measures)
            measure_width = self.content_width / self.measures_per_line

            # 1. 6선 보강선 (Staff Lines)
            for s in range(6):
                sy = y_cursor + (s * 20)
                ET.SubElement(svg, "line", {
                    "x1": str(self.margin),
                    "y1": str(sy),
                    "x2": str(self.margin + (m_count * measure_width)),
                    "y2": str(sy),
                    "stroke": "#555555",
                    "stroke-width": "1.5"
                })

            # 2. 마디선 및 음표 렌더링
            for m_idx, m_data in enumerate(line_measures):
                m_x = self.margin + (m_idx * measure_width)

                # 시작/끝 마디선
                ET.SubElement(svg, "line", {
                    "x1": str(m_x), "y1": str(y_cursor),
                    "x2": str(m_x), "y2": str(y_cursor + 100),
                    "stroke": "#000000", "stroke-width": "2"
                })
                if m_idx == m_count - 1:
                    ET.SubElement(svg, "line", {
                        "x1": str(m_x + measure_width), "y1": str(y_cursor),
                        "x2": str(m_x + measure_width), "y2": str(y_cursor + 100),
                        "stroke": "#000000", "stroke-width": "2"
                    })

                # 마디 번호
                m_num_elem = ET.SubElement(svg, "text", {
                    "x": str(m_x + 5),
                    "y": str(y_cursor - 8),
                    "font-family": "sans-serif",
                    "font-size": "14",
                    "font-weight": "bold",
                    "fill": "#1D4ED8"
                })
                m_num_elem.text = str(m_data.get("number"))

                # 음표/프렛 이벤트
                events = m_data.get("events", [])
                for ev in events:
                    string_num = ev.get("string")
                    fret_val = ev.get("fret")
                    pos_str = ev.get("position", "0/1")

                    if string_num is not None and 1 <= string_num <= 6:
                        # 1번 줄이 맨 위 (y_cursor)
                        ey = y_cursor + ((string_num - 1) * 20)
                        
                        # 위치 계산 (0/1 -> 0.1, 1/4 -> 0.35, 1/2 -> 0.6 등)
                        try:
                            num, den = map(float, pos_str.split("/"))
                            frac = num / den if den != 0 else 0.0
                        except Exception:
                            frac = 0.0
                        
                        ex = m_x + 30 + (frac * (measure_width - 50))

                        # 프렛 배경 원
                        ET.SubElement(svg, "circle", {
                            "cx": str(ex), "cy": str(ey), "r": "9",
                            "fill": "#FFFFFF", "stroke": "#000000", "stroke-width": "1"
                        })

                        # 프렛 숫자 (null인 경우 ?)
                        txt_val = str(fret_val) if fret_val is not None else "?"
                        f_elem = ET.SubElement(svg, "text", {
                            "x": str(ex), "y": str(ey + 4),
                            "font-family": "sans-serif",
                            "font-size": "12",
                            "font-weight": "bold",
                            "text-anchor": "middle",
                            "fill": "#000000" if fret_val is not None else "#DC2626"
                        })
                        f_elem.text = txt_val

            y_cursor += self.line_spacing

        tree = ET.ElementTree(svg)
        tree.write(output_svg_path, encoding="utf-8", xml_declaration=True)
        return output_svg_path

    def render_pdf(self, svg_path: str, output_pdf_path: str = "output/score.pdf") -> bool:
        """SVG 파일을 PDF로 변환합니다. (cairosvg 활용)"""
        try:
            import cairosvg
            cairosvg.svg2pdf(url=svg_path, write_to=output_pdf_path)
            return True
        except Exception as e:
            print(f"⚠️ CairoSVG 기반 PDF 변환 실패: {e}")
            return False
