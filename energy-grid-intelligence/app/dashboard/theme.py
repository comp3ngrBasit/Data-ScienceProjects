"""Design system for the dashboard: palette, CSS, Plotly template and reusable UI components.

Categorical series colours were validated for the dark card surface (#151F32) with the dataviz
palette validator (lightness band, chroma, adjacent-pair CVD separation, contrast): fixed order, never cycled.
Status colours (good / warning / critical) are reserved for state and always ship with a text label.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG, BG2, CARD = "#080D18", "#101827", "#151F32"
TEXT, TEXT2, GRID = "#F8FAFC", "#94A3B8", "#24314A"
BLUE, CYAN, GREEN, AMBER, RED = "#3B82F6", "#22D3EE", "#22C55E", "#F59E0B", "#EF4444"  # brand + status
SERIES = ["#3B82F6", "#C9780A", "#0E9FB8", "#8B6CEF", "#D9488F", "#5FA83A"]  # validated categorical order
SEQ = ["#0B1A33", "#12325E", "#1A4C8C", "#2465BA", "#3B82F6", "#7DB0F9", "#C3DAFC"]  # one hue, dark -> light
SEVERITY = {"critical": RED, "high": "#F97316", "medium": AMBER, "low": TEXT2}

pio.templates["egi"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor=CARD, plot_bgcolor=CARD, colorway=SERIES,
    font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", color=TEXT2, size=12),
    title=dict(font=dict(color=TEXT, size=14), x=0.01, xanchor="left"),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT2), orientation="h", y=1.08, x=0),
    hoverlabel=dict(bgcolor=BG2, bordercolor=GRID, font=dict(color=TEXT)),
    margin=dict(l=48, r=18, t=48, b=40), hovermode="closest"))
pio.templates.default = "egi"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background: {BG}; }}
section[data-testid="stSidebar"] {{ background: {BG2}; border-right: 1px solid {GRID}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }}
h1, h2, h3 {{ color: {TEXT}; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.55rem !important; font-weight: 650 !important; }}
.egi-sub {{ color: {TEXT2}; font-size: 0.9rem; margin: -0.6rem 0 1.2rem 0; }}
.egi-card {{ background: {CARD}; border: 1px solid {GRID}; border-radius: 12px; padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(0,0,0,.35); height: 100%; transition: border-color .15s ease; }}
.egi-card:hover {{ border-color: #33415C; }}
.egi-kpi-label {{ color: {TEXT2}; font-size: 0.74rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.egi-kpi-value {{ color: {TEXT}; font-size: 1.6rem; font-weight: 650; margin-top: 6px; line-height: 1.15; }}
.egi-kpi-unit {{ color: {TEXT2}; font-size: 0.85rem; font-weight: 500; margin-left: 4px; }}
.egi-kpi-note {{ color: {TEXT2}; font-size: 0.76rem; margin-top: 6px; }}
.egi-accent {{ width: 28px; height: 3px; border-radius: 2px; margin-bottom: 10px; }}
.egi-callout {{ background: {BG2}; border: 1px solid {GRID}; border-left: 3px solid {BLUE}; border-radius: 8px;
               padding: 10px 14px; color: {TEXT2}; font-size: 0.86rem; margin: 6px 0 14px 0; }}
.egi-callout b {{ color: {TEXT}; }}
.egi-badge {{ display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: .72rem; font-weight: 600;
             border: 1px solid {GRID}; color: {TEXT}; }}
div[data-testid="stPlotlyChart"] {{ background: {CARD}; border: 1px solid {GRID}; border-radius: 12px; padding: 6px; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {GRID}; border-radius: 10px; }}
button[data-baseweb="tab"] {{ font-weight: 550; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(f"# {html.escape(title)}")
    st.markdown(f'<div class="egi-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, unit: str = "", note: str = "", accent: str = BLUE) -> str:
    return (f'<div class="egi-card"><div class="egi-accent" style="background:{accent}"></div>'
            f'<div class="egi-kpi-label">{html.escape(label)}</div>'
            f'<div class="egi-kpi-value">{html.escape(value)}<span class="egi-kpi-unit">{html.escape(unit)}</span></div>'
            f'<div class="egi-kpi-note">{html.escape(note)}</div></div>')


def kpi_row(items: list[tuple], cols: int | None = None) -> None:
    columns = st.columns(cols or len(items))
    for c, it in zip(columns, items):
        c.markdown(kpi(*it), unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


def callout(text_html: str, color: str = BLUE) -> None:
    st.markdown(f'<div class="egi-callout" style="border-left-color:{color}">{text_html}</div>', unsafe_allow_html=True)


def fmt(x, nd: int = 1, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{x:,.{nd}f}" + (" %" if pct else "")
    except (TypeError, ValueError):
        return str(x)


def style(fig: go.Figure, height: int = 380, title: str | None = None, **kw) -> go.Figure:
    fig.update_layout(height=height, title=title, **kw)
    return fig


def missing(what: str) -> None:
    st.warning(f"{what} is not available yet. Run `python -m app.pipeline run` to compute it.", icon="⚠️")
