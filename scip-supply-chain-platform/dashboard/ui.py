"""Shared dashboard styling (teal enterprise theme), KPI cards and chart helpers."""
from __future__ import annotations

import html

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BRAND, BRAND_DARK, BRAND_LIGHT, TRACK = "#2f9e8a", "#1f6f61", "#a9dccf", "#eeeeee"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"Critical": "#d03b3b", "Warning": "#fab219", "Info": "#2a78d6", "Healthy": "#0ca30c", "Reorder": "#fab219",
          "Overstock": "#2a78d6", "High": "#d03b3b", "Medium": "#fab219", "Low": "#0ca30c", "Overdue": "#d03b3b",
          "At risk": "#fab219", "On track": "#0ca30c", "Delivered": "#898781", "Strategic": "#0ca30c",
          "Preferred": BRAND, "Watch": "#fab219"}
GRID = "#e6efec"

CSS = """
<style>
:root {--teal:#2f9e8a; --teal-dark:#1f6f61; --teal-deep:#17574c;}
.block-container {padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1560px;}
section[data-testid="stSidebar"] > div:first-child {background: linear-gradient(180deg,#bfe6db 0%,#a5d9ca 55%,#8ccdbb 100%);}
section[data-testid="stSidebar"] .brand {font-weight:800; font-size:1.15rem; color:var(--teal-deep); text-align:center; margin:4px 0 0}
section[data-testid="stSidebar"] .tag {text-align:center; color:#2c5d54; font-size:.76rem; margin-bottom:10px}
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] {gap:9px; width:100%}
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {width:100%}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {width:100%; display:flex; justify-content:center;
  background:#e9f7f3; border:1.5px solid #5fae9e; border-radius:22px; padding:8px 12px; margin:0;
  box-shadow:0 2px 0 rgba(23,87,76,.25); cursor:pointer}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {display:none}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {font-weight:650; color:var(--teal-deep); font-size:.86rem}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {background:#fff}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {
  background:linear-gradient(180deg,#ffffff,#cdeee5); border-color:var(--teal-dark); border-width:2px; box-shadow:0 3px 0 rgba(23,87,76,.45)}
.ribbon-wrap {display:flex; justify-content:center; margin:0 0 14px}
.ribbon {background:linear-gradient(180deg,#3aa892,#23796a); color:#fff; font-weight:800; font-size:1.4rem; padding:8px 70px 10px;
  clip-path:polygon(0 0,100% 0,95% 100%,5% 100%); text-align:center; box-shadow:0 4px 10px rgba(0,0,0,.15)}
.ribbon small {display:block; font-weight:500; font-size:.76rem; opacity:.92}
div[class*="st-key-panel"] {background:#fff; border-radius:16px !important; border:1px solid rgba(23,87,76,.10) !important;
  box-shadow:0 3px 10px rgba(23,87,76,.10)}
.panel-title {font-weight:700; color:#16352f; font-size:.95rem; margin:2px 0 4px}
.ikpi {display:flex; align-items:center; gap:12px; background:#fff; border-radius:16px; padding:12px 14px;
  box-shadow:0 3px 10px rgba(23,87,76,.12); border:1px solid rgba(23,87,76,.08); min-height:86px}
.ikpi .ico {font-size:1.6rem; width:46px; height:46px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  background:#e3f4ef; flex-shrink:0}
.ikpi .val {font-size:1.45rem; font-weight:800; color:var(--teal-deep); line-height:1.1; white-space:nowrap}
.ikpi .lbl {font-size:.78rem; color:#3d6b62; font-weight:600}
.ikpi .sub {font-size:.7rem; color:#6b8a84}
.badge {display:inline-block; padding:1px 9px; border-radius:999px; font-size:.75rem; font-weight:650}
.origin {background:#fff8e6; border:1px solid #f0d78c; border-radius:10px; padding:6px 12px; font-size:.8rem; color:#5c4a00; margin-bottom:.7rem}
.rec {border:1px solid rgba(23,87,76,.12); border-radius:12px; padding:9px 13px; margin-bottom:7px; background:#fff}
.rec .t {font-weight:650} .rec .w {color:#52514e; font-size:.84rem; margin-top:3px}
</style>
"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)
    t = go.layout.Template()
    t.layout = go.Layout(font=dict(family="system-ui, Segoe UI, sans-serif", size=12, color="#52514e"), colorway=SERIES,
                         paper_bgcolor="#fff", plot_bgcolor="#fff", margin=dict(l=10, r=10, t=40, b=10),
                         title=dict(font=dict(size=14, color="#0b0b0b"), x=0),
                         xaxis=dict(showgrid=False, linecolor="#c3c2b7"), yaxis=dict(gridcolor=GRID, zeroline=False),
                         legend=dict(orientation="h", yanchor="top", y=-0.15, x=0), hoverlabel=dict(bgcolor="white"))
    pio.templates["scip"] = t
    pio.templates.default = "plotly_white+scip"


def ribbon(title: str, sub: str = ""):
    s = f"<small>{html.escape(sub)}</small>" if sub else ""
    st.markdown(f"<div class='ribbon-wrap'><div class='ribbon'>{html.escape(title)}{s}</div></div>", unsafe_allow_html=True)


def origin_note(text: str):
    st.markdown(f"<div class='origin'>🧾 {text}</div>", unsafe_allow_html=True)


def ikpi(icon, value, label, sub=""):
    return (f"<div class='ikpi'><div class='ico'>{icon}</div><div><div class='val'>{html.escape(str(value))}</div>"
            f"<div class='lbl'>{html.escape(label)}</div><div class='sub'>{html.escape(sub)}</div></div></div>")


def kpi_row(cards):
    for c, h in zip(st.columns(len(cards)), cards):
        c.markdown(h, unsafe_allow_html=True)


def panel(key: str):
    return st.container(border=True, key=f"panel_{key}")


def title(text: str):
    st.markdown(f"<div class='panel-title'>{html.escape(text)}</div>", unsafe_allow_html=True)


def chart(fig, key: str):
    fig.update_layout(template="plotly_white+scip", paper_bgcolor="#fff", plot_bgcolor="#fff")
    st.plotly_chart(fig, theme=None, key=f"chart_{key}", config={"displaylogo": False})


def badge(level: str) -> str:
    c = STATUS.get(level, "#898781")
    return f"<span class='badge' style='background:{c}22;color:{c};border:1px solid {c}55'>{html.escape(level)}</span>"


def money(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"${v / 1e6:,.2f}M" if abs(v) >= 1e6 else f"${v / 1e3:,.1f}K" if abs(v) >= 1e4 else f"${v:,.0f}"


def num(v, d=0):
    return "n/a" if v is None else f"{v:,.{d}f}"


def pct(v, d=1):
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def table(df, height="auto", **kw):
    st.dataframe(df, width="stretch", hide_index=True, height=height, **kw)
