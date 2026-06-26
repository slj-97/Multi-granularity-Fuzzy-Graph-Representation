from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch


OUT = Path(__file__).resolve().parent

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "legend.frameon": False,
    }
)

BLUE = "#3B6EA8"
ORANGE = "#D9892B"
GREEN = "#5A9E6F"
GRAY = "#6E6E6E"
LIGHT_BLUE = "#DCE8F5"
LIGHT_ORANGE = "#F3E3CA"


V = np.array(
    [
        [5 / 3, 3 / 2, 3 / 2, 1, 1, 1],
        [3 / 2, 5 / 3, 3 / 2, 1, 1, 1],
        [3 / 2, 3 / 2, 5 / 3, 1, 1, 1],
        [1, 1, 1, 5 / 3, 3 / 2, 3 / 2],
        [1, 1, 1, 3 / 2, 5 / 3, 3 / 2],
        [1, 1, 1, 3 / 2, 3 / 2, 5 / 3],
    ],
    dtype=float,
)


def save(fig, stem):
    for suffix, kwargs in {
        ".pdf": {},
        ".png": {"dpi": 600},
        ".svg": {},
    }.items():
        fig.savefig(OUT / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def style_full_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("black")
    ax.tick_params(direction="in", length=3, width=0.8, colors="black", top=True, right=True)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_weight("bold")


def rounded(ax, xy, width, height, color, label=None, lw=0.9, ls="-"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw,
        edgecolor=color,
        facecolor="white",
        linestyle=ls,
    )
    ax.add_patch(box)
    if label:
        ax.text(xy[0] + width / 2, xy[1] + height / 2, label, ha="center", va="center")


def fig_granular_layers():
    fig, ax = plt.subplots(figsize=(2.35, 1.75))
    ax.set_xlim(0.25, 6.75)
    ax.set_ylim(-0.35, 2.85)
    ax.axis("off")

    xs = np.arange(1, 7)
    for x in xs:
        ax.plot([x, x], [0.55, 2.22], color="#B8B8B8", lw=0.6, ls=(0, (3, 3)), zorder=0)

    for x in xs:
        rounded(ax, (x - 0.25, 2.22), 0.5, 0.34, GRAY, rf"$x_{x}$")

    rounded(ax, (0.58, 1.15), 2.84, 0.46, BLUE, r"$\{x_1,x_2,x_3\}$", lw=1.0)
    rounded(ax, (3.58, 1.15), 2.84, 0.46, ORANGE, r"$\{x_4,x_5,x_6\}$", lw=1.0)
    rounded(ax, (0.58, 0.05), 5.84, 0.46, GRAY, r"$U=\{x_1,\ldots,x_6\}$", lw=1.0)

    ax.text(0.33, 2.39, r"$\lambda=0.95$", ha="right", va="center")
    ax.text(0.33, 1.38, r"$\lambda=0.85$", ha="right", va="center")
    ax.text(0.33, 0.28, r"$\lambda=0.70$", ha="right", va="center")
    ax.text(3.5, 2.72, "Nested granular layers", ha="center", va="center", weight="bold")
    save(fig, "fig7_granular")


def fig_vector_components():
    components = np.arange(1, 7)
    series = {
        "MG-FGR": V[0],
        "Single-coarse": np.ones(6),
        "Single-medium": np.array([1, 1, 1, 0, 0, 0], dtype=float),
        "Single-fine": np.array([1, 0, 0, 0, 0, 0], dtype=float),
    }
    colors = {
        "MG-FGR": BLUE,
        "Single-coarse": GRAY,
        "Single-medium": GREEN,
        "Single-fine": ORANGE,
    }

    fig, ax = plt.subplots(figsize=(2.35, 1.75))
    for name, values in series.items():
        ax.plot(
            components,
            values,
            marker="o",
            lw=1.2,
            ms=3.3,
            color=colors[name],
            label=name,
        )

    ax.set_xlabel("Component index")
    ax.set_ylabel("Value")
    ax.set_xticks(components)
    ax.set_ylim(-0.08, 1.78)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 5 / 3])
    ax.set_yticklabels(["0", "0.5", "1.0", "1.5", r"$5/3$"])
    ax.grid(axis="y", color="#E2E2E2", lw=0.6)
    style_full_box(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        fontsize=5.5,
        handlelength=1.3,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    save(fig, "fig8_vector")


def pca_two_dimensional(X):
    centered = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def add_group_ellipse(ax, pts, color):
    center = pts.mean(axis=0)
    width = max(pts[:, 0].ptp() + 0.18, 0.18)
    height = max(pts[:, 1].ptp() + 0.18, 0.18)
    ax.add_patch(
        Ellipse(
            center,
            width=width,
            height=height,
            edgecolor=color,
            facecolor=color,
            alpha=0.14,
            lw=0.9,
        )
    )


def fig_embedding():
    coords = pca_two_dimensional(V)
    if coords[:3, 0].mean() > coords[3:, 0].mean():
        coords[:, 0] *= -1

    fig, ax = plt.subplots(figsize=(2.35, 1.75))
    groups = [
        (slice(0, 3), BLUE, LIGHT_BLUE, r"$\{x_1,x_2,x_3\}$"),
        (slice(3, 6), ORANGE, LIGHT_ORANGE, r"$\{x_4,x_5,x_6\}$"),
    ]
    for sl, color, _, label in groups:
        pts = coords[sl]
        add_group_ellipse(ax, pts, color)
        ax.scatter(pts[:, 0], pts[:, 1], s=25, color=color, edgecolor="white", linewidth=0.5, label=label, zorder=3)

    for i, (x, y) in enumerate(coords, start=1):
        ax.text(x + 0.025, y + 0.015, rf"$x_{i}$", fontsize=6.5)

    ax.axhline(0, color="#DDDDDD", lw=0.6, zorder=0)
    ax.axvline(0, color="#DDDDDD", lw=0.6, zorder=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    style_full_box(ax)
    ax.set_title("PCA of MG-FGR vectors", fontsize=7, weight="bold", pad=3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6, handletextpad=0.4)
    save(fig, "fig9_embedding")


def main():
    assert np.allclose(V[0], [5 / 3, 3 / 2, 3 / 2, 1, 1, 1])
    fig_granular_layers()
    fig_vector_components()
    fig_embedding()
    for stem in ("fig7_granular", "fig8_vector", "fig9_embedding"):
        assert (OUT / f"{stem}.pdf").exists()


if __name__ == "__main__":
    main()
