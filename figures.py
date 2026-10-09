"""Milestone 7: interactive Plotly figures for the dashboard.

Colors follow the dataviz skill's reference palette, with separate light and dark steps
(validated with its palette checker in both modes):
  blue    this KOI's transits / its binned fold   (categorical slot 1)
  orange  other KOIs' transits on the same star   (categorical slot 2)
  gray    ordinary data points, kept recessive so the signal stands out
"""
import numpy as np
import plotly.graph_objects as go

from inference import KoiResult
from search_transits import binned_median, fold_hours

FOLD_WINDOW = 3.0   # fold plot spans +/- this many transit durations
FOLD_BINS_PER_DURATION = 10

THEMES = {
    "light": {"blue": "#2a78d6", "orange": "#eb6834", "points": "#898781",
              "text": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"blue": "#3987e5", "orange": "#d95926", "points": "#898781",
             "text": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835"},
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def ppm(y: np.ndarray) -> np.ndarray:
    return (y - 1) * 1e6


def y_range(y: np.ndarray, signal: np.ndarray, clip_pct: float = 0.05) -> list[float]:
    """Y limits that ignore the most extreme clip_pct% of points at each end but always show the full dip."""
    lo = min(np.percentile(y, clip_pct), np.min(signal) if len(signal) else 0)
    hi = np.percentile(y, 100 - clip_pct)
    pad = 0.08 * (hi - lo)
    return [lo - pad, hi + pad]


def layout(fig: go.Figure, c: dict, x_title: str, y_title: str, height: int) -> go.Figure:
    axis = dict(gridcolor=c["grid"], gridwidth=1, linecolor=c["axis"], zeroline=False, automargin=True,
                tickfont=dict(color=c["text"]), title_font=dict(color=c["text"]))
    fig.update_layout(
        height=height, margin=dict(l=70, r=10, t=40, b=55),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=c["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        hovermode="closest",
    )
    fig.update_xaxes(title_text=x_title, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig


def points(x, y, name: str, color: str, size: float, opacity: float, hover: str) -> go.Scattergl:
    return go.Scattergl(x=x, y=y, mode="markers", name=name,
                        marker=dict(color=color, size=size, opacity=opacity, line_width=0),
                        hovertemplate=hover + "<extra></extra>")


def light_curve_figure(r: KoiResult, theme: str = "light") -> go.Figure:
    """The whole cleaned light curve, with this KOI's (and other KOIs') transits highlighted."""
    c = THEMES[theme]
    t, y = r.star_t, ppm(r.star_y)

    def in_transit(period, t0, duration):
        return np.abs(fold_hours(t, period, t0)) < duration * 24 / 2

    this = in_transit(r.period, r.t0, r.duration)
    other = np.zeros(len(t), dtype=bool)
    for name, ephemeris in r.ephemerides.items():
        if name != r.kepoi_name:
            other |= in_transit(*ephemeris)
    other &= ~this
    rest = ~(this | other)

    hover = "%{x:.3f} BKJD<br>%{y:.0f} ppm"
    fig = go.Figure()
    fig.add_trace(points(t[rest], y[rest], "Out of transit", c["points"], 2.5, 0.5, hover))
    if other.any():
        fig.add_trace(points(t[other], y[other], "Other KOIs' transits", c["orange"], 4, 0.9, hover))
    fig.add_trace(points(t[this], y[this], f"{r.kepoi_name} transits", c["blue"], 4, 0.9, hover))
    dips = y[this | other]
    fig.update_yaxes(range=y_range(y[rest], np.percentile(dips, [1]) if len(dips) else dips))
    return layout(fig, c, "Time (BKJD = days since 2009-01-01)", "Brightness change (ppm)", 380)


def fold_figure(r: KoiResult, theme: str = "light") -> go.Figure:
    """Every transit stacked on top of each other (folded on the period), plus the binned median."""
    c = THEMES[theme]
    duration_h = r.duration * 24
    x, y = fold_hours(r.t, r.period, r.t0), ppm(r.y)
    window = np.abs(x) < FOLD_WINDOW * duration_h
    x, y = x[window], y[window]
    bx, by = binned_median(x, y, duration_h / FOLD_BINS_PER_DURATION)

    fig = go.Figure()
    fig.add_vrect(x0=-duration_h / 2, x1=duration_h / 2, fillcolor=c["blue"], opacity=0.10,
                  line_width=0, layer="below")
    fig.add_trace(points(x, y, "Individual measurements", c["points"], 2.5, 0.25,
                         "%{x:.2f} h<br>%{y:.0f} ppm"))
    # Scattergl, not Scatter: WebGL traces draw above SVG ones, so an SVG line would be hidden
    fig.add_trace(go.Scattergl(x=bx, y=by, mode="lines", name="Binned median",
                             line=dict(color=c["blue"], width=2),
                             hovertemplate="%{x:.2f} h<br>median %{y:.0f} ppm<extra></extra>"))
    fig.update_layout(annotations=[dict(
        x=0, y=1, xref="x", yref="paper", yanchor="bottom", showarrow=False,
        text=f"catalog duration {duration_h:.1f} h", font=dict(size=11, color=c["text"]))])
    fig.update_yaxes(range=y_range(y, by, clip_pct=0.5))  # the fold is dense: clip harder
    return layout(fig, c, "Hours from mid-transit", "Brightness change (ppm)", 380)
