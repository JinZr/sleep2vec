from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_ASSET_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"

_TEXT_COLOR = "#171717"
_AXIS_COLOR = "#1E1E1E"
_GRID_COLOR = "#E8EBF2"
_FIGURE_BG = "#FFFFFF"
_AXES_BG = "#FFFFFF"

_PRIMARY_DARK = "#43508F"
_PRIMARY = "#5E71BE"
_PRIMARY_MID = "#7889D1"
_PRIMARY_LIGHT = "#AEB9EA"
_PRIMARY_PALE = "#D8DFF5"

_OPENAI_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "openai_like_eval_blues",
    ["#F7F8FC", "#E2E7F7", "#B7C2ED", "#7B8CD1", "#43508F"],
)


@lru_cache(maxsize=1)
def register_local_fonts() -> None:
    for font_path in (
        _ASSET_FONT_DIR / "roboto_mono" / "RobotoMono-VariableFont_wght.ttf",
        _ASSET_FONT_DIR / "roboto_mono" / "RobotoMono-Italic-VariableFont_wght.ttf",
        _ASSET_FONT_DIR / "jetbrains_mono" / "JetBrainsMono-VariableFont_wght.ttf",
        _ASSET_FONT_DIR / "jetbrains_mono" / "JetBrainsMono-Italic-VariableFont_wght.ttf",
        _ASSET_FONT_DIR / "inter" / "Inter-VariableFont_opsz,wght.ttf",
        _ASSET_FONT_DIR / "inter" / "Inter-Italic-VariableFont_opsz,wght.ttf",
    ):
        if font_path.is_file():
            fm.fontManager.addfont(str(font_path))


@lru_cache(maxsize=1)
def pick_font_family() -> str:
    register_local_fonts()
    preferred = [
        "Roboto Mono",
        "JetBrains Mono",
        "Inter",
        "Aptos",
        "Arial",
        "Helvetica Neue",
        "Helvetica",
        "DejaVu Sans",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Sans"


def use_openai_like_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": _FIGURE_BG,
            "axes.facecolor": _AXES_BG,
            "savefig.facecolor": _FIGURE_BG,
            "savefig.edgecolor": _FIGURE_BG,
            "savefig.dpi": 300,
            "figure.dpi": 180,
            "font.family": pick_font_family(),
            "font.size": 11,
            "text.color": _TEXT_COLOR,
            "axes.labelcolor": _TEXT_COLOR,
            "axes.edgecolor": _AXIS_COLOR,
            "axes.linewidth": 1.15,
            "axes.titlesize": 18,
            "axes.titleweight": 700,
            "axes.titlelocation": "center",
            "axes.labelsize": 14,
            "axes.labelweight": 400,
            "xtick.color": _TEXT_COLOR,
            "ytick.color": _TEXT_COLOR,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": False,
        }
    )


def style_axes(ax: plt.Axes, *, show_grid: bool) -> None:
    ax.set_facecolor(_AXES_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_AXIS_COLOR)
    ax.spines["bottom"].set_color(_AXIS_COLOR)
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)
    ax.tick_params(axis="both", colors=_TEXT_COLOR, labelsize=11, pad=8)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.set_axisbelow(True)
    if show_grid:
        ax.grid(True, color=_GRID_COLOR, linestyle=(0, (6, 8)), linewidth=0.9, alpha=1.0)
    else:
        ax.grid(False)


__all__ = [
    "pick_font_family",
    "register_local_fonts",
    "style_axes",
    "use_openai_like_theme",
    "_TEXT_COLOR",
    "_FIGURE_BG",
    "_AXES_BG",
    "_PRIMARY_DARK",
    "_PRIMARY",
    "_PRIMARY_MID",
    "_PRIMARY_LIGHT",
    "_PRIMARY_PALE",
    "_OPENAI_BLUE_CMAP",
]
