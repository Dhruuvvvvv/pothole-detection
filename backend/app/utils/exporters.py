"""CSV and PDF inspection report generators for RoadGuard AI."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.app.models.db_models import AnalysisRecord, DetectionRecord
from ml.assessment.severity_engine import METHODOLOGY_DISCLAIMER


def generate_analysis_csv_bytes(
    analysis: AnalysisRecord,
    detections: List[DetectionRecord],
) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "analysis_id",
        "filename",
        "analysis_type",
        "timestamp_utc",
        "detection_index",
        "frame_index",
        "track_id",
        "class_id",
        "class_code",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "box_width_px",
        "box_height_px",
        "box_area_px",
        "image_area_pct",
        "priority_score",
        "priority_band",
    ])

    for idx, d in enumerate(detections, start=1):
        writer.writerow([
            analysis.id,
            analysis.filename,
            analysis.analysis_type,
            analysis.timestamp.isoformat() if analysis.timestamp else "",
            idx,
            d.frame_index if d.frame_index is not None else "",
            d.track_id if d.track_id is not None else "",
            d.class_id,
            d.class_code,
            d.class_name,
            f"{d.confidence:.4f}",
            f"{d.x1:.1f}",
            f"{d.y1:.1f}",
            f"{d.x2:.1f}",
            f"{d.y2:.1f}",
            f"{d.box_width:.1f}",
            f"{d.box_height:.1f}",
            f"{d.box_area:.1f}",
            f"{d.image_area_pct:.3f}",
            f"{d.priority_score:.2f}",
            d.priority_band,
        ])
    return buf.getvalue().encode("utf-8")


def generate_analysis_pdf_bytes(
    analysis: AnalysisRecord,
    detections: List[DetectionRecord],
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"RoadGuard AI Report #{analysis.id}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "RGTitle",
        parent=styles["Heading1"],
        fontSize=17,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "RGSub",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    sec_style = ParagraphStyle(
        "RGSection",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=6,
    )
    note_style = ParagraphStyle(
        "RGNote",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=colors.HexColor("#334155"),
        leading=12,
    )

    story = [
        Paragraph("RoadGuard AI — Automated Road Surface Inspection Report", title_style),
        Paragraph(
            f"Report ID: #{analysis.id} &nbsp;|&nbsp; File: {analysis.filename} &nbsp;|&nbsp; "
            f"Modality: {analysis.analysis_type.upper()} &nbsp;|&nbsp; "
            f"Timestamp: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') if analysis.timestamp else 'N/A'}",
            sub_style,
        ),
    ]

    # Summary KPI table
    summary_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Status", analysis.status.upper(), "Processing Time", f"{analysis.processing_duration:.3f} s"],
        ["Total Detections", str(analysis.total_detections), "High Priority", str(analysis.high_priority_count)],
        ["Medium Priority", str(analysis.medium_priority_count), "Low Priority", str(analysis.low_priority_count)],
    ]
    if analysis.summary_json:
        try:
            s_obj = json.loads(analysis.summary_json)
            if analysis.analysis_type == "video":
                summary_data.append([
                    "Processed Frames",
                    str(s_obj.get("processed_frames", "N/A")),
                    "Local Track IDs",
                    str(s_obj.get("unique_track_ids_observed", "N/A")),
                ])
        except Exception:
            pass

    kpi_table = Table(summary_data, colWidths=[42 * mm, 45 * mm, 42 * mm, 45 * mm])
    kpi_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(kpi_table)
    story.append(Spacer(1, 8))

    # Embed annotated image preview when modality is image and file exists
    if analysis.analysis_type == "image" and analysis.output_path:
        out_p = Path(analysis.output_path)
        if out_p.is_file() and out_p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            story.append(Paragraph("Annotated Inspection Visualization", sec_style))
            img_flow = RLImage(str(out_p), width=155 * mm, height=95 * mm, kind="proportional")
            story.append(img_flow)
            story.append(Spacer(1, 8))

    story.append(Paragraph("Detected Road Damage Instances (Top 50 by Visual Priority)", sec_style))
    if not detections:
        story.append(Paragraph("No road damage instances exceeded the active confidence threshold.", note_style))
    else:
        det_rows = [["#", "Code", "Category", "Conf", "Box (x1,y1,x2,y2)", "Area %", "Score", "Priority"]]
        for i, d in enumerate(detections[:50], start=1):
            det_rows.append([
                str(i),
                d.class_code,
                d.class_name,
                f"{d.confidence * 100:.1f}%",
                f"({d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f})",
                f"{d.image_area_pct:.2f}%",
                f"{d.priority_score:.1f}",
                d.priority_band,
            ])
        det_table = Table(
            det_rows,
            colWidths=[10 * mm, 14 * mm, 36 * mm, 16 * mm, 42 * mm, 18 * mm, 16 * mm, 22 * mm],
            repeatRows=1,
        )
        det_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(det_table)

    story.append(Spacer(1, 12))
    story.append(Paragraph("Engineering Methodology & Calibration Note", sec_style))
    story.append(Paragraph(METHODOLOGY_DISCLAIMER, note_style))

    doc.build(story)
    return buf.getvalue()
