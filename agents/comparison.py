# agents/comparison.py — Final comparison of the 3 agents

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from core.utils import setup_logger
from config import OUTPUT_DIR, OUTPUT_COMPARISON, RENDER_DPI

logger = setup_logger()


def run_comparison(results: list) -> str:
    """
    Generate a professional comparison chart for the 3 agents.

    Args:
        results: List of 3 result dicts from DreamerXL, XDreamer, ATD.

    Returns:
        Path to the saved comparison PNG.
    """
    logger.info("Comparison — generating final comparison chart...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    names       = [r["agent"]      for r in results]
    methods     = [r["method"]     for r in results]
    confidences = [r["confidence"] for r in results]

    # Agent-specific metrics (normalized to [0,1])
    metric1 = []  # TSM / AMA final loss inverted / avg grounding
    metric2 = []  # SDXL CFG normalized / domain gap inverted / path efficiency
    metric3 = []  # object detected / lora rank normalized / waypoints normalized

    for r in results:
        if r["agent"] == "Dreamer XL":
            metric1.append(r["tsm_score"])
            metric2.append(min(r["sdxl_guidance"] / 12.0, 1.0))
            metric3.append(0.9 if r["object"] != "main 3D object" else 0.4)
        elif r["agent"] == "X-Dreamer":
            metric1.append(max(0, 1 - r["ama_loss"]))
            metric2.append(max(0, 1 - r["domain_gap"]))
            metric3.append(min(r["lora_rank"] / 64.0, 1.0))
        elif r["agent"] == "ATD":
            metric1.append(r["avg_grounding"])
            metric2.append(r["path_efficiency"])
            metric3.append(min(r["waypoints"] / 10.0, 1.0))

    colors = ["#00d4ff", "#ff6b6b", "#00ff88"]
    bg = "#0a0a1a"

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(bg)

    # ── 1. Confidence bar chart ───────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor(bg)
    bars = ax1.bar(names, [c * 100 for c in confidences], color=colors, alpha=0.85, width=0.5)
    for bar, conf in zip(bars, confidences):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{conf:.0%}", ha="center", color="white", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.set_title("Confidence Score (%)", color="white", fontsize=11, fontweight="bold")
    ax1.tick_params(colors="white")
    ax1.set_facecolor(bg)
    for s in ax1.spines.values(): s.set_color("white"); s.set_alpha(0.2)

    # ── 2. Radar chart ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2, polar=True)
    ax2.set_facecolor(bg)
    categories = ["Confidence", "Metric 1\n(Quality)", "Metric 2\n(Alignment)", "Metric 3\n(Structure)"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for i, (r, color) in enumerate(zip(results, colors)):
        values = [confidences[i], metric1[i], metric2[i], metric3[i]]
        values += values[:1]
        ax2.plot(angles, values, color=color, linewidth=2, label=names[i])
        ax2.fill(angles, values, color=color, alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, color="white", fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_title("Multi-Criteria Radar", color="white", fontsize=11,
                  fontweight="bold", pad=20)
    ax2.tick_params(colors="white")
    ax2.set_facecolor("#0a0a1a")
    ax2.spines["polar"].set_color("white")
    ax2.spines["polar"].set_alpha(0.2)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
               fontsize=8, facecolor=bg, labelcolor="white")

    # ── 3. Method comparison table ────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor(bg)
    ax3.set_xticks([]); ax3.set_yticks([])
    for s in ax3.spines.values(): s.set_visible(False)
    ax3.set_title("Agent Methods", color="white", fontsize=11, fontweight="bold")

    table_data = []
    for r, color in zip(results, colors):
        table_data.append([r["agent"], r["method"].replace(" + ", "\n+ ")])

    y = 0.85
    for (agent, method), color in zip(table_data, colors):
        ax3.text(0.02, y, f"● {agent}", color=color, fontsize=10,
                 fontweight="bold", transform=ax3.transAxes)
        ax3.text(0.08, y - 0.08, method, color="white", fontsize=8,
                 transform=ax3.transAxes, alpha=0.85)
        y -= 0.28

    # ── 4. Per-metric grouped bar chart ──────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor(bg)
    x = np.arange(3)
    width = 0.22
    ax4.bar(x - width, metric1, width, label="Quality / TSM / Grounding", color=colors[0], alpha=0.8)
    ax4.bar(x,         metric2, width, label="Alignment / CFG / Efficiency", color=colors[1], alpha=0.8)
    ax4.bar(x + width, metric3, width, label="Structure / Rank / Waypoints", color=colors[2], alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, color="white", fontsize=9)
    ax4.set_ylim(0, 1.2)
    ax4.set_title("Per-Metric Performance", color="white", fontsize=11, fontweight="bold")
    ax4.tick_params(colors="white")
    ax4.legend(fontsize=7, facecolor=bg, labelcolor="white", loc="upper right")
    for s in ax4.spines.values(): s.set_color("white"); s.set_alpha(0.2)

    # ── 5. Ranking ────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor(bg)
    ranked = sorted(zip(confidences, names, colors), reverse=True)
    medals = ["#1", "#2", "#3"]
    ax5.set_xticks([]); ax5.set_yticks([])
    for s in ax5.spines.values(): s.set_visible(False)
    ax5.set_title("Final Ranking", color="white", fontsize=11, fontweight="bold")
    for i, (conf, name, color) in enumerate(ranked):
        ax5.text(0.15, 0.75 - i*0.28, f"{medals[i]}  {name}",
                 transform=ax5.transAxes, color=color,
                 fontsize=14, fontweight="bold")
        ax5.text(0.15, 0.75 - i*0.28 - 0.1, f"     Confidence: {conf:.0%}",
                 transform=ax5.transAxes, color="white", fontsize=9)

    # ── 6. Summary text ───────────────────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor(bg)
    ax6.set_xticks([]); ax6.set_yticks([])
    for s in ax6.spines.values(): s.set_visible(False)
    ax6.set_title("Pipeline Summary", color="white", fontsize=11, fontweight="bold")

    best = ranked[0]
    summary = (
        f"Best agent   : {best[1]}\n"
        f"Confidence   : {best[0]:.0%}\n\n"
        f"Dreamer XL   : Text → 3D (TSM + SDXL)\n"
        f"X-Dreamer    : 2D → 3D (CG-LoRA + AMA)\n"
        f"ATD          : Language → Navigation\n\n"
        f"All 3 agents simulate the\nDreamer World Family pipeline\nwithout GPU."
    )
    ax6.text(0.08, 0.5, summary, transform=ax6.transAxes,
             color="white", fontsize=9, va="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#111122", alpha=0.7))

    fig.suptitle("DREAMER WORLD FAMILY — 3 AGENTS COMPARISON",
                 color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_COMPARISON, dpi=RENDER_DPI, facecolor=bg)
    plt.close()

    logger.info(f"Comparison — saved to: {OUTPUT_COMPARISON}")
    return OUTPUT_COMPARISON
