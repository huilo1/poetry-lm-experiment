from __future__ import annotations

import html
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / "article" / "poetry_lm_experiment_report.html"
PDF_PATH = ROOT / "article" / "poetry_lm_experiment_report.pdf"


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ),
        (
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        ),
    ]
    for regular, bold, mono in candidates:
        if all(Path(p).exists() for p in (regular, bold, mono)):
            pdfmetrics.registerFont(TTFont("ArticleSerif", regular))
            pdfmetrics.registerFont(TTFont("ArticleSerifBold", bold))
            pdfmetrics.registerFont(TTFont("ArticleMono", mono))
            return "ArticleSerif", "ArticleSerifBold", "ArticleMono"
    raise FileNotFoundError("No suitable Cyrillic fonts found for PDF export")


def inline_html(node: Tag) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(html.escape(str(child)))
            continue
        if not isinstance(child, Tag):
            continue
        inner = inline_html(child)
        if child.name in {"strong", "b"}:
            parts.append(f"<b>{inner}</b>")
        elif child.name in {"em", "i"}:
            parts.append(f"<i>{inner}</i>")
        elif child.name == "code":
            parts.append(f"<font face='ArticleMono'>{inner}</font>")
        elif child.name == "br":
            parts.append("<br/>")
        else:
            parts.append(inner)
    return "".join(parts)


def clean_text(node: Tag) -> str:
    return " ".join(part for part in node.get_text("\n", strip=True).splitlines() if part.strip())


def build_styles(font_regular: str, font_bold: str, font_mono: str):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ArticleBody",
            parent=styles["BodyText"],
            fontName=font_regular,
            fontSize=10.5,
            leading=15,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ArticleLead",
            parent=styles["BodyText"],
            fontName=font_regular,
            fontSize=11.2,
            leading=16,
            spaceAfter=10,
            textColor=colors.HexColor("#5b6473"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="ArticleH1",
            parent=styles["Heading1"],
            fontName=font_bold,
            fontSize=21,
            leading=24,
            spaceAfter=12,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ArticleH2",
            parent=styles["Heading2"],
            fontName=font_bold,
            fontSize=15,
            leading=19,
            spaceBefore=16,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ArticleH3",
            parent=styles["Heading3"],
            fontName=font_bold,
            fontSize=12.5,
            leading=16,
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ArticleSmall",
            parent=styles["BodyText"],
            fontName=font_regular,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#5b6473"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ArticleCode",
            parent=styles["Code"],
            fontName=font_mono,
            fontSize=8.8,
            leading=11.5,
            spaceAfter=8,
        )
    )
    return styles


def table_from_html(table_tag: Tag, styles) -> Table:
    rows: list[list[str]] = []
    for tr in table_tag.find_all("tr"):
        row = []
        for cell in tr.find_all(["th", "td"]):
            row.append(clean_text(cell))
        if row:
            rows.append(row)
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "ArticleSerif"),
                ("FONTNAME", (0, 0), (-1, 0), "ArticleSerifBold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3ede3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#374151")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8cdbf")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def sample_card_to_story(card: Tag, styles) -> list:
    story: list = []
    title = card.find("h4")
    if title:
        story.append(Paragraph(inline_html(title), styles["ArticleH3"]))
    for node in card.children:
        if not isinstance(node, Tag):
            continue
        if node.name == "h4":
            continue
        if node.name == "div":
            story.append(Paragraph(inline_html(node), styles["ArticleSmall"]))
        elif node.name == "pre":
            story.append(Preformatted(node.get_text(), styles["ArticleCode"]))
    story.append(Spacer(1, 4))
    return story


def html_to_story(html_text: str):
    font_regular, font_bold, font_mono = register_fonts()
    styles = build_styles(font_regular, font_bold, font_mono)
    soup = BeautifulSoup(html_text, "html.parser")
    article = soup.find("article", class_="paper")
    if article is None:
        raise ValueError("article body not found")

    story: list = []
    for node in article.children:
        if isinstance(node, NavigableString):
            continue
        if not isinstance(node, Tag):
            continue
        if node.name == "h1":
            story.append(Paragraph(inline_html(node), styles["ArticleH1"]))
        elif node.name == "h2":
            story.append(Paragraph(inline_html(node), styles["ArticleH2"]))
        elif node.name in {"h3", "h4"}:
            story.append(Paragraph(inline_html(node), styles["ArticleH3"]))
        elif node.name == "p":
            style = styles["ArticleLead"] if "lead" in (node.get("class") or []) else styles["ArticleBody"]
            style = styles["ArticleSmall"] if "small" in (node.get("class") or []) else style
            story.append(Paragraph(inline_html(node), style))
        elif node.name == "div" and "meta" in (node.get("class") or []):
            for card in node.find_all("div", class_="meta-card", recursive=False):
                story.append(Paragraph(f"<b>{inline_html(card.find('strong'))}</b> {html.escape(clean_text(card).replace(clean_text(card.find('strong')), '', 1).strip())}", styles["ArticleBody"]))
            story.append(Spacer(1, 4))
        elif node.name == "div" and "box" in (node.get("class") or []):
            story.append(Paragraph(inline_html(node), styles["ArticleSmall"]))
        elif node.name == "div" and "figure" in (node.get("class") or []):
            caption = node.find("div", class_="caption")
            if caption:
                story.append(Paragraph(f"<i>{inline_html(caption)}</i>", styles["ArticleSmall"]))
            else:
                story.append(Paragraph("<i>График см. в HTML-версии статьи.</i>", styles["ArticleSmall"]))
        elif node.name == "table":
            story.append(table_from_html(node, styles))
            story.append(Spacer(1, 8))
        elif node.name == "ul":
            items = []
            for li in node.find_all("li", recursive=False):
                items.append(ListItem(Paragraph(inline_html(li), styles["ArticleBody"])))
            story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=14))
            story.append(Spacer(1, 6))
        elif node.name == "div" and "sample-grid" in (node.get("class") or []):
            for card in node.find_all("div", class_="sample-card", recursive=False):
                story.extend(sample_card_to_story(card, styles))
        else:
            text = clean_text(node)
            if text:
                story.append(Paragraph(html.escape(text), styles["ArticleBody"]))
    return story


def main() -> None:
    html_text = HTML_PATH.read_text(encoding="utf-8")
    story = html_to_story(html_text)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Обучение с нуля русской поэтической модели продолжения стихов",
        author="OpenAI Codex",
    )
    doc.build(story)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
