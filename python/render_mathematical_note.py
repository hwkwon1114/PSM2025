"""Render the trajectory-learning mathematical note as a vector PDF.

The workspace does not provide a LaTeX engine.  This renderer intentionally uses only
Matplotlib, which is already available in the project environment.  Markdown prose is
laid out as vector text and display equations are rendered by Matplotlib's mathtext.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "reports" / "trajectory_information_world_model_math.md"
OUTPUT = ROOT / "docs" / "reports" / "trajectory_information_world_model_math.pdf"
PREVIEW = ROOT / "docs" / "reports" / "trajectory_information_world_model_math_preview.png"

TITLE = "Mathematical Formulation of\nTrajectory-Aware Multi-Fidelity Learning"
SUBTITLE = (
    "Predictive representations, neural operators, active experiment design,\n"
    "and world-model learning for history-dependent forming"
)


def sanitize_mathtext(text: str) -> str:
    return (
        text.replace(r"\boldsymbol", "")
        .replace(r"\bigl", r"\left")
        .replace(r"\bigr", r"\right")
        .replace(r"\Bigl", r"\left")
        .replace(r"\Bigr", r"\right")
    )


def clean_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = text.replace("**", "").replace("`", "")
    return sanitize_mathtext(text)


class NoteRenderer:
    def __init__(self, pdf: PdfPages, preview_path: Path, header: str):
        self.pdf = pdf
        self.preview_path = preview_path
        self.header = header
        self.fig = None
        self.y = 0.0
        self.page_number = 0
        self.preview_written = False
        plt.rcParams.update(
            {
                "font.family": "DejaVu Serif",
                "mathtext.fontset": "stix",
                "axes.unicode_minus": False,
            }
        )

    def _new_page(self, content: bool = True) -> None:
        if self.fig is not None:
            self._finish_page()
        self.page_number += 1
        self.fig = plt.figure(figsize=(8.5, 11.0), facecolor="white")
        self.fig.subplots_adjust(0, 0, 1, 1)
        if content:
            self.fig.text(
                0.08,
                0.968,
                self.header,
                fontsize=7.2,
                color="#51606f",
                va="top",
                weight="bold",
            )
            self.fig.lines.append(
                plt.Line2D(
                    [0.08, 0.92],
                    [0.948, 0.948],
                    transform=self.fig.transFigure,
                    color="#9ba7b4",
                    linewidth=0.7,
                )
            )
            self.y = 0.925
        else:
            self.y = 0.93

    def _finish_page(self) -> None:
        if self.fig is None:
            return
        self.fig.text(
            0.5,
            0.026,
            str(self.page_number),
            fontsize=7.5,
            color="#66717d",
            ha="center",
        )
        if self.page_number == 4 and not self.preview_written:
            self.fig.savefig(self.preview_path, dpi=150, bbox_inches=None)
            self.preview_written = True
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    def ensure(self, height: float) -> None:
        if self.fig is None:
            self._new_page()
        if self.y - height < 0.065:
            self._new_page()

    def title_page(
        self,
        title: str,
        subtitle: str,
        project_label: str,
        date_label: str,
    ) -> None:
        self._new_page(content=False)
        self.fig.text(
            0.10,
            0.79,
            title,
            fontsize=25,
            weight="bold",
            color="#172b4d",
            va="top",
            linespacing=1.2,
        )
        self.fig.text(
            0.10,
            0.60,
            subtitle,
            fontsize=12.5,
            color="#40556d",
            va="top",
            linespacing=1.45,
        )
        self.fig.lines.append(
            plt.Line2D(
                [0.10, 0.54],
                [0.55, 0.55],
                transform=self.fig.transFigure,
                color="#1f77b4",
                linewidth=2.2,
            )
        )
        self.fig.text(
            0.10,
            0.49,
            date_label,
            fontsize=10.5,
            color="#51606f",
        )
        self.fig.text(
            0.10,
            0.15,
            project_label,
            fontsize=9.5,
            color="#6b7785",
            linespacing=1.4,
        )

    def contents_page(self, sections: list[str]) -> None:
        self._new_page(content=True)
        self.heading("Contents", level=1)
        for index, section in enumerate(sections, start=1):
            self.ensure(0.034)
            self.fig.text(
                0.115,
                self.y,
                f"{index:02d}",
                fontsize=9.2,
                color="#1f77b4",
                weight="bold",
                va="top",
            )
            self.fig.text(
                0.17,
                self.y,
                section,
                fontsize=10.0,
                color="#243447",
                va="top",
            )
            self.y -= 0.038

    def heading(self, text: str, level: int) -> None:
        text = clean_markdown(text)
        if level == 1:
            self.ensure(0.085)
            if self.y < 0.82:
                self._new_page()
            size, color, gap = 16.0, "#172b4d", 0.063
        elif level == 2:
            self.ensure(0.065)
            size, color, gap = 12.4, "#1f4e79", 0.050
        else:
            self.ensure(0.052)
            size, color, gap = 10.6, "#334e68", 0.041
        self.fig.text(
            0.09,
            self.y,
            text,
            fontsize=size,
            color=color,
            weight="bold",
            va="top",
        )
        self.y -= gap

    def paragraph(self, text: str) -> None:
        text = clean_markdown(text.strip())
        if not text:
            return
        lines = textwrap.wrap(
            text,
            width=100,
            break_long_words=False,
            break_on_hyphens=False,
        )
        height = 0.0205 * len(lines) + 0.012
        self.ensure(height)
        self.fig.text(
            0.09,
            self.y,
            "\n".join(lines),
            fontsize=9.25,
            color="#202b36",
            va="top",
            linespacing=1.34,
        )
        self.y -= height

    def bullet(self, text: str, marker: str = "•") -> None:
        text = clean_markdown(text.strip())
        lines = textwrap.wrap(
            text,
            width=93,
            subsequent_indent="   ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        height = 0.020 * len(lines) + 0.005
        self.ensure(height)
        self.fig.text(
            0.105,
            self.y,
            marker,
            fontsize=9.2,
            color="#1f77b4",
            va="top",
            weight="bold",
        )
        self.fig.text(
            0.132,
            self.y,
            "\n".join(lines),
            fontsize=9.05,
            color="#283746",
            va="top",
            linespacing=1.3,
        )
        self.y -= height

    def quote(self, text: str) -> None:
        text = clean_markdown(text.strip())
        lines = textwrap.wrap(text, width=88, break_long_words=False)
        height = 0.022 * len(lines) + 0.022
        self.ensure(height)
        self.fig.patches.append(
            plt.Rectangle(
                (0.095, self.y - height + 0.010),
                0.006,
                height,
                transform=self.fig.transFigure,
                facecolor="#4c9fd6",
                edgecolor="none",
            )
        )
        self.fig.text(
            0.12,
            self.y,
            "\n".join(lines),
            fontsize=9.4,
            color="#294b66",
            style="italic",
            va="top",
            linespacing=1.35,
        )
        self.y -= height

    def equation(self, equation: str) -> None:
        equation = sanitize_mathtext(equation.strip())
        # Matplotlib mathtext supports \left/\right but not all LaTeX delimiter-size
        # aliases (notably \bigl and \bigr in the project environment).
        equation = (
            equation.replace(r"\bigl", r"\left")
            .replace(r"\bigr", r"\right")
            .replace(r"\Bigl", r"\left")
            .replace(r"\Bigr", r"\right")
        )
        height = 0.070 if len(equation) < 115 else 0.082
        self.ensure(height)
        fontsize = 11.3
        if len(equation) > 150:
            fontsize = 8.5
        elif len(equation) > 115:
            fontsize = 9.5
        self.fig.text(
            0.5,
            self.y - 0.010,
            f"${equation}$",
            fontsize=fontsize,
            color="#101820",
            ha="center",
            va="top",
        )
        self.y -= height

    def close(self) -> None:
        self._finish_page()


def parse_and_render(renderer: NoteRenderer, source: str) -> None:
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "## Abstract")
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            renderer.paragraph(" ".join(part.strip() for part in paragraph))
            paragraph.clear()

    for raw in lines[start:]:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("$$") and stripped.endswith("$$"):
            flush()
            renderer.equation(stripped[2:-2])
            continue
        heading = re.match(r"^(#{2,4})\s+(.*)$", stripped)
        if heading:
            flush()
            renderer.heading(heading.group(2), level=len(heading.group(1)) - 1)
            continue
        if stripped.startswith(">"):
            flush()
            renderer.quote(stripped[1:].strip())
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if numbered:
            flush()
            renderer.bullet(numbered.group(2), marker=numbered.group(1) + ".")
            continue
        if stripped.startswith("- "):
            flush()
            renderer.bullet(stripped[2:])
            continue
        paragraph.append(stripped)
    flush()


def render_document(
    source_path: Path,
    output_path: Path,
    preview_path: Path,
    title: str,
    subtitle: str,
    sections: list[str],
    header: str,
    subject: str,
    keywords: str,
    project_label: str,
    date_label: str = "Research note · 20 August 2026",
) -> None:
    source = source_path.read_text(encoding="utf-8")
    metadata = {
        "Title": title.replace("\n", " "),
        "Author": "PSM2025-zigzag research project",
        "Subject": subject,
        "Keywords": keywords,
    }
    with PdfPages(output_path, metadata=metadata) as pdf:
        renderer = NoteRenderer(pdf, preview_path=preview_path, header=header)
        renderer.title_page(
            title=title,
            subtitle=subtitle,
            project_label=project_label,
            date_label=date_label,
        )
        renderer.contents_page(sections)
        parse_and_render(renderer, source)
        renderer.close()
    print(f"Wrote {output_path}")
    print(f"Wrote {preview_path}")


def main() -> None:
    sections = [
        "Controlled history-dependent system",
        "Path space, measures, and initial sampling",
        "When can a high-dimensional state be compressed?",
        "Neural-operator predictive world model",
        "Representation failure and diagnostics",
        "Bayesian information acquisition",
        "Ergodic trajectory synthesis",
        "Multi-fidelity calibration",
        "Closed-loop active world-model algorithm",
        "Validation hypotheses",
        "Evaluation metrics and falsification tests",
        "Recommended experimental sequence",
        "Research gap and references",
    ]
    render_document(
        source_path=SOURCE,
        output_path=OUTPUT,
        preview_path=PREVIEW,
        title=TITLE,
        subtitle=SUBTITLE,
        sections=sections,
        header="TRAJECTORY-AWARE MULTI-FIDELITY LEARNING",
        subject="Trajectory-aware multi-fidelity predictive world models",
        keywords="neural operators, JEPA, active learning, experiment design",
        project_label="PSM2025-zigzag\nEnglish-wheel surrogate modeling project",
    )


if __name__ == "__main__":
    main()
