# agents/atd_agent.py — Agent 3: ATD (Adaptive Text Dreamer)
# Simulates: Language imagination for robot navigation

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from core.utils import setup_logger
from config import OUTPUT_DIR, OUTPUT_ATD, RENDER_DPI

logger = setup_logger()


class ATDAgent:
    """
    Agent 3 — ATD: Adaptive Text Dreamer Simulation
    =================================================
    Real paper: Adaptive Text Dreamer (Zhang et al., 2025)

    Core concepts simulated:
    - Language Imagination: the robot agent reads a text instruction and
      generates an internal imagined trajectory of the environment before
      physically navigating it. This allows zero-shot navigation in
      unseen environments.
    - Adaptive Planning: the navigation plan adapts at each waypoint
      based on language grounding (matching text tokens to spatial regions).
    - Waypoint generation: key positions the robot must visit, derived
      from the text instruction via language-conditioned imagination.

    This agent:
      1. Parses the prompt for navigation-relevant keywords.
      2. Generates a set of imagined waypoints on a 2D navigation map.
      3. Simulates language grounding scores per waypoint.
      4. Renders a top-down navigation map with the imagined path.
      5. Outputs: atd_navmap.png + confidence score.
    """

    NAVIGATION_KEYWORDS = {
        "go":      "move forward",
        "find":    "search target",
        "reach":   "approach goal",
        "explore": "explore area",
        "enter":   "enter zone",
        "avoid":   "obstacle avoidance",
        "follow":  "follow path",
        "cross":   "cross boundary",
        "turn":    "change direction",
        "stop":    "halt at target",
    }

    OBJECT_KEYWORDS = {
        "robot": "robot", "dragon": "dragon", "castle": "castle",
        "car": "car", "spaceship": "spaceship", "warrior": "warrior",
        "city": "city", "forest": "forest", "space": "space",
        "mountain": "mountain", "house": "house",
    }

    def run(self, prompt: str) -> dict:
        """Run the ATD language imagination + navigation pipeline."""
        logger.info("ATD — starting language imagination for navigation...")
        lower = prompt.lower()

        target_object   = self._detect_target(lower)
        nav_instructions = self._extract_instructions(lower)
        waypoints        = self._generate_waypoints(nav_instructions, prompt)
        grounding_scores = self._compute_grounding_scores(waypoints, prompt)
        path_efficiency  = self._compute_path_efficiency(waypoints)
        confidence       = self._compute_confidence(target_object, grounding_scores, path_efficiency)

        output_path = self._render(target_object, waypoints, grounding_scores, nav_instructions, path_efficiency)

        result = {
            "agent":            "ATD",
            "method":           "Language Imagination + Robot Navigation",
            "prompt":           prompt,
            "target":           target_object,
            "instructions":     nav_instructions,
            "waypoints":        len(waypoints),
            "avg_grounding":    round(float(np.mean(grounding_scores)), 3),
            "path_efficiency":  path_efficiency,
            "confidence":       confidence,
            "output":           output_path,
        }

        logger.info(f"ATD — done. Confidence: {confidence:.0%} | Waypoints: {len(waypoints)}")
        self._print_output(result)
        return result

    # ── Private ─────────────────────────────────────────────────────────────

    def _detect_target(self, prompt: str) -> str:
        for k, v in self.OBJECT_KEYWORDS.items():
            if k in prompt:
                return v
        return "target object"

    def _extract_instructions(self, prompt: str) -> list:
        found = []
        for kw, instruction in self.NAVIGATION_KEYWORDS.items():
            if kw in prompt:
                found.append(instruction)
        if not found:
            found = ["explore area", "approach goal", "halt at target"]
        return found

    def _generate_waypoints(self, instructions: list, prompt: str) -> list:
        """
        Generate imagined waypoints on a 10x10 navigation grid.
        Number of waypoints scales with instruction complexity.
        """
        n = max(4, len(instructions) + 3 + len(prompt.split()) // 5)
        n = min(n, 10)
        random.seed(sum(ord(c) for c in prompt))
        waypoints = [(1.0, 1.0)]  # start
        for _ in range(n - 2):
            x = random.uniform(1, 9)
            y = random.uniform(1, 9)
            waypoints.append((round(x, 2), round(y, 2)))
        waypoints.append((9.0, 9.0))  # goal
        return waypoints

    def _compute_grounding_scores(self, waypoints: list, prompt: str) -> list:
        """
        Language grounding score per waypoint: how well the text
        instruction aligns with the imagined spatial position.
        """
        scores = []
        words = prompt.lower().split()
        for i, (x, y) in enumerate(waypoints):
            base = 0.5 + (i / len(waypoints)) * 0.3
            noise = random.gauss(0, 0.05)
            word_bonus = 0.1 if i < len(words) and len(words[i % len(words)]) > 4 else 0
            scores.append(round(min(1.0, max(0.1, base + noise + word_bonus)), 3))
        return scores

    def _compute_path_efficiency(self, waypoints: list) -> float:
        """
        Path efficiency = straight-line distance / actual path length.
        1.0 = perfectly straight, 0.0 = very inefficient.
        """
        if len(waypoints) < 2:
            return 1.0
        straight = np.sqrt((waypoints[-1][0]-waypoints[0][0])**2 +
                           (waypoints[-1][1]-waypoints[0][1])**2)
        total = sum(
            np.sqrt((waypoints[i+1][0]-waypoints[i][0])**2 +
                    (waypoints[i+1][1]-waypoints[i][1])**2)
            for i in range(len(waypoints)-1)
        )
        return round(float(straight / total), 3) if total > 0 else 1.0

    def _compute_confidence(self, target, grounding, efficiency) -> float:
        score = 0.3
        if target != "target object":
            score += 0.25
        score += float(np.mean(grounding)) * 0.25
        score += efficiency * 0.2
        return round(min(score, 1.0), 2)

    def _render(self, target, waypoints, grounding, instructions, efficiency) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        palette = {"bg": "#0a1628", "accent": "#00ff88", "path": "#ffcc00", "robot": "#00aaff"}

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.patch.set_facecolor(palette["bg"])

        # ── Left: Navigation map ─────────────────────────────────────────
        ax = axes[0]
        ax.set_facecolor(palette["bg"])
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(palette["accent"]); s.set_alpha(0.3)
        ax.set_title("Imagined Navigation Map", color=palette["accent"], fontsize=13, fontweight="bold")

        # Grid
        for i in range(1, 10):
            ax.axhline(i, color="white", alpha=0.05, lw=0.5)
            ax.axvline(i, color="white", alpha=0.05, lw=0.5)

        # Random obstacles
        random.seed(42)
        for _ in range(8):
            ox, oy = random.uniform(1.5, 8.5), random.uniform(1.5, 8.5)
            ax.add_patch(plt.Rectangle((ox, oy), 0.6, 0.6,
                                       color="#ff4444", alpha=0.4, zorder=2))

        # Path
        xs = [w[0] for w in waypoints]
        ys = [w[1] for w in waypoints]
        ax.plot(xs, ys, color=palette["path"], linewidth=2, linestyle="--",
                alpha=0.7, zorder=3)

        # Waypoints with grounding score
        for i, ((x, y), score) in enumerate(zip(waypoints, grounding)):
            color = palette["robot"] if i == 0 else (palette["accent"] if i == len(waypoints)-1 else palette["path"])
            ax.scatter([x], [y], s=120, color=color, zorder=5, edgecolors="white", linewidths=0.8)
            ax.text(x+0.15, y+0.15, f"{score:.2f}", fontsize=7, color="white", alpha=0.85, zorder=6)

        # Start / Goal labels
        ax.text(xs[0]-0.1, ys[0]-0.4, "START", fontsize=8, color=palette["robot"],
                fontweight="bold", zorder=6)
        ax.text(xs[-1]-0.1, ys[-1]+0.2, "GOAL", fontsize=8, color=palette["accent"],
                fontweight="bold", zorder=6)

        # Robot icon (circle with arrow)
        ax.add_patch(plt.Circle((xs[0], ys[0]), 0.35, color=palette["robot"], alpha=0.3, zorder=4))

        # Target label
        ax.text(5, 0.3, f"Target: {target.upper()}", ha="center", fontsize=9,
                color=palette["accent"], alpha=0.8)

        # ── Right: Grounding scores + stats ─────────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor(palette["bg"])
        for s in ax2.spines.values(): s.set_color(palette["accent"]); s.set_alpha(0.3)

        # Grounding bar chart
        wp_labels = [f"WP{i}" for i in range(len(waypoints))]
        bar_colors = [palette["robot"]] + [palette["path"]]*(len(waypoints)-2) + [palette["accent"]]
        bars = ax2.bar(wp_labels, grounding, color=bar_colors, alpha=0.8, zorder=3)
        ax2.set_ylim(0, 1.15)
        ax2.axhline(y=np.mean(grounding), color="white", linestyle="--",
                    alpha=0.5, lw=1.2, zorder=4)
        ax2.text(len(waypoints)-1, np.mean(grounding)+0.03,
                 f"avg={np.mean(grounding):.2f}", color="white", fontsize=8, ha="right")

        ax2.set_title("Language Grounding Scores per Waypoint",
                      color=palette["accent"], fontsize=11, fontweight="bold")
        ax2.set_xlabel("Waypoint", color="white", fontsize=9)
        ax2.set_ylabel("Grounding Score", color="white", fontsize=9)
        ax2.tick_params(colors="white")

        # Instructions panel
        instr_text = "\n".join([f"• {instr}" for instr in instructions[:5]])
        ax2.text(0.02, 0.02,
                 f"Instructions:\n{instr_text}\n\nPath Efficiency: {efficiency:.2f}",
                 transform=ax2.transAxes, color="white", fontsize=8.5,
                 va="bottom", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#1a2a1a", alpha=0.6))

        fig.suptitle(f"AGENT 3 — ATD  ·  Language Imagination  ·  Target: {target.title()}",
                     color=palette["accent"], fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_ATD, dpi=RENDER_DPI, facecolor=palette["bg"])
        plt.close()
        return OUTPUT_ATD

    def _print_output(self, r: dict) -> None:
        print(f"\n    Model           : {r['agent']}")
        print(f"    Method          : {r['method']}")
        print(f"    Target          : {r['target']}")
        print(f"    Instructions    : {', '.join(r['instructions'])}")
        print(f"    Waypoints       : {r['waypoints']}")
        print(f"    Avg Grounding   : {r['avg_grounding']}")
        print(f"    Path Efficiency : {r['path_efficiency']}")
        print(f"    Confidence      : {r['confidence']:.0%}")
        print(f"    Output          : {r['output']}")
