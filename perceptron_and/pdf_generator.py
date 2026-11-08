"""Generación del informe PDF del entrenamiento del perceptrón."""

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether
)


def create_training_pdf(result, initial_values, learning_rate, max_epochs):
    """Crea un PDF en memoria y devuelve sus bytes para st.download_button."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(letter),
        rightMargin=1.2*cm, leftMargin=1.2*cm,
        topMargin=1.2*cm, bottomMargin=1.2*cm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="MainTitle", parent=styles["Title"], alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"), spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name="SmallText", parent=styles["BodyText"], fontSize=7,
        leading=9, spaceAfter=3
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading2"],
        textColor=colors.HexColor("#1F4E79"), spaceBefore=10, spaceAfter=6
    ))

    story = []
    story.append(Paragraph("Informe de entrenamiento: Perceptrón AND", styles["MainTitle"]))
    story.append(Paragraph(
        "Proyecto práctico de Inteligencia Artificial / Ciencia de Datos",
        styles["Heading3"]
    ))
    story.append(Spacer(1, 8))

    summary = [
        ["Parámetro", "Valor"],
        ["Pesos iniciales", f"w1 = {initial_values['w1']:.4f} | w2 = {initial_values['w2']:.4f}"],
        ["Bias inicial", f"{initial_values['bias']:.4f}"],
        ["Tasa de aprendizaje (eta)", f"{learning_rate:.4f}"],
        ["Máximo de épocas configurado", str(max_epochs)],
        ["Épocas ejecutadas", str(result["epochs_completed"])],
        ["Pesos finales", f"w1 = {result['w1']:.4f} | w2 = {result['w2']:.4f}"],
        ["Bias final", f"{result['bias']:.4f}"],
        ["Estado", "Convergió (cero errores)" if result["converged"] else "No convergió dentro del límite"],
    ]
    t = Table(summary, colWidths=[6*cm, 11*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#EAF0F6")),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)

    story.append(Paragraph("1. Tabla resumen por épocas", styles["Section"]))
    epoch_data = [["Época", "w1 al final", "w2 al final", "Bias al final", "Errores"]]
    for row in result["history"]:
        epoch_data.append([
            str(row["epoch"]), f"{row['w1']:.4f}", f"{row['w2']:.4f}",
            f"{row['bias']:.4f}", str(row["errors"])
        ])
    et = Table(epoch_data, repeatRows=1)
    et.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("PADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(et)

    story.append(Paragraph("2. Procedimiento matemático paso a paso", styles["Section"]))
    story.append(Paragraph(
        "Para cada patrón se calcula z = (x1 × w1) + (x2 × w2) + b. "
        "La función escalón produce y = 1 cuando z ≥ 0; en otro caso y = 0. "
        "El error es e = objetivo − y. Después se actualiza cada parámetro: "
        "w1 nuevo = w1 anterior + eta × e × x1; "
        "w2 nuevo = w2 anterior + eta × e × x2; "
        "b nuevo = b anterior + eta × e.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 6))

    detail_data = [[
        "Época", "Muestra", "(x1,x2)", "Objetivo", "Pesos/bias antes",
        "z", "y", "e", "Cambios (Δw1, Δw2, Δb)", "Pesos/bias después"
    ]]
    for s in result["steps"]:
        detail_data.append([
            str(s["epoch"]), str(s["sample"]), f"({s['x1']},{s['x2']})",
            str(s["target"]),
            f"{s['w1_before']:.3f}, {s['w2_before']:.3f}, {s['bias_before']:.3f}",
            f"{s['z']:.3f}", str(s["prediction"]), str(s["error"]),
            f"{s['delta_w1']:.3f}, {s['delta_w2']:.3f}, {s['delta_bias']:.3f}",
            f"{s['w1_after']:.3f}, {s['w2_after']:.3f}, {s['bias_after']:.3f}",
        ])

    detail_table = Table(
        detail_data, repeatRows=1,
        colWidths=[1.0*cm, 1.1*cm, 1.4*cm, 1.2*cm, 3.0*cm,
                   1.1*cm, 0.8*cm, 0.8*cm, 3.2*cm, 3.0*cm]
    )
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 6),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("PADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(detail_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "3. Resultado esperado de la compuerta AND",
        styles["Section"]
    ))
    truth = [
        ["x1", "x2", "Salida esperada"],
        ["0", "0", "0"], ["0", "1", "0"],
        ["1", "0", "0"], ["1", "1", "1"]
    ]
    tt = Table(truth, colWidths=[2*cm, 2*cm, 4*cm])
    tt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("PADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(tt)

    doc.build(story)
    return buffer.getvalue()
