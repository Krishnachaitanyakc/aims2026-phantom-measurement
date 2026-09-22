#!/usr/bin/env python3
"""Vector plot of observed Haiku parser sensitivity; no inference calls."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
cells = json.loads((ROOT / "evidence/camera_ready_evidence.json").read_text())[
    "models"
]["haiku"]["cells"]
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), constrained_layout=True)
depths = [1, 5, 7]
for ax, metric, title in zip(
    axes, ["D", "acc"], ["Extracted answer agreement", "Correct final answers"]
):
    for construction, label, color, marker in [
        ("degenerate", "Omitted, permissive", "#D55E00", "o"),
        ("fixed", "Repeated", "#0072B2", "s"),
    ]:
        values = [cells[f"{construction}_k{k}"][metric] for k in depths]
        ax.plot(
            depths,
            values,
            color=color,
            marker=marker,
            label=label,
            linewidth=1.3,
            markersize=4,
        )
    if metric == "D":
        ax.plot(
            depths,
            [cells[f"degenerate_k{k}"]["D_marker_only"] for k in depths],
            color="#222222",
            marker="^",
            linestyle="--",
            label="Omitted, marker only",
            linewidth=1.2,
            markersize=4,
        )
    ax.set(
        title=title,
        xlabel="Calls per chain",
        ylim=(-0.04, 1.06),
        xticks=depths,
        yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
    )
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(loc="lower left", fontsize=6, frameon=False)
axes[1].set_ylabel("Fraction of 60 responses")
(ROOT / "paper/figures").mkdir(exist_ok=True)
fig.savefig(
    ROOT / "paper/figures/parser-sensitivity.pdf",
    metadata={"CreationDate": None, "ModDate": None},
)
plt.close(fig)
