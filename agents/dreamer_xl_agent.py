# agents/dreamer_xl_agent.py — Agent 1: Dreamer XL
# Simulates: Text → 3D via Trajectory Score Matching + Stable Diffusion XL

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from core.utils import setup_logger
from config import OUTPUT_DIR, OUTPUT_DREAMER_XL, RENDER_DPI, ENVIRONMENT_PALETTES

logger = setup_logger()


class DreamerXLAgent:
    """
    Agent 1 — Dreamer XL Simulation
    ================================
    Real paper: Dreamer XL (Miao et al., 2024)

    Core concepts simulated:
    - Trajectory Score Matching (TSM): optimizes the 3D object along a
      trajectory of diffusion timesteps, ensuring multi-view consistency.
    - Stable Diffusion XL (SDXL): used as a 2D prior to guide 3D generation
      via Score Distillation Sampling (SDS).

    This agent:
      1. Parses the text prompt into a structured 3D scene description.
      2. Simulates TSM by computing a "trajectory score" over multiple steps.
      3. Renders a styled 3D-looking matplotlib scene.
      4. Outputs: dreamer_xl_scene.png + confidence score.
    """

    OBJECT_KEYWORDS = {
        "robot": "robot", "dragon": "dragon", "castle": "medieval castle",
        "car": "car", "spaceship": "spaceship", "warrior": "armored warrior",
        "crystal": "crystal", "wolf": "wolf", "tree": "ancient tree",
        "city": "urban city", "house": "house", "panda": "giant panda",
        "volcano": "volcano", "lighthouse": "lighthouse", "submarine": "submarine",
    }

    ENVIRONMENT_KEYWORDS = {
        "city": "futuristic city", "mountain": "mountain landscape",
        "forest": "forest", "space": "outer space", "ocean": "underwater scene",
        "desert": "desert landscape", "room": "interior room",
    }

    def run(self, prompt: str) -> dict:
        """
        Run the Dreamer XL pipeline on the given prompt.

        Returns a result dict with: output_path, confidence, tsm_score,
        object, environment, sdxl_guidance.
        """
        logger.info("DreamerXL — starting Text→3D pipeline...")
        lower = prompt.lower()

        main_object  = self._detect_object(lower)
        environment  = self._detect_environment(lower)
        tsm_score    = self._simulate_tsm(prompt)
        sdxl_guidance = self._compute_sdxl_guidance(main_object, environment)
        confidence   = self._compute_confidence(main_object, environment, tsm_score)

        palette = ENVIRONMENT_PALETTES.get(environment, ENVIRONMENT_PALETTES["simple digital environment"])
        output_path = self._render(main_object, environment, palette, tsm_score, sdxl_guidance)

        result = {
            "agent":        "Dreamer XL",
            "method":       "Trajectory Score Matching + SDXL",
            "prompt":       prompt,
            "object":       main_object,
            "environment":  environment,
            "tsm_score":    tsm_score,
            "sdxl_guidance": sdxl_guidance,
            "confidence":   confidence,
            "output":       output_path,
        }

        logger.info(f"DreamerXL — done. Confidence: {confidence:.0%} | TSM: {tsm_score:.2f}")
        self._print_output(result)
        return result

    # ── Private ─────────────────────────────────────────────────────────────

    def _detect_object(self, prompt: str) -> str:
        for k, v in self.OBJECT_KEYWORDS.items():
            if k in prompt:
                return v
        return "main 3D object"

    def _detect_environment(self, prompt: str) -> str:
        for k, v in self.ENVIRONMENT_KEYWORDS.items():
            if k in prompt:
                return v
        return "simple digital environment"

    def _simulate_tsm(self, prompt: str) -> float:
        """
        Simulate Trajectory Score Matching over T diffusion timesteps.
        TSM minimizes the score gap between consecutive trajectory steps.
        Returns a normalized score in [0.0, 1.0].
        """
        T = 20  # timesteps
        trajectory = []
        base = len(prompt) % 7 + 1
        for t in range(T):
            noise = random.gauss(0, 0.05)
            score = 1.0 - (1.0 / (1 + base * np.exp(-0.3 * t))) + noise
            trajectory.append(max(0.0, min(1.0, score)))
        return round(float(np.mean(trajectory[-5:])), 3)  # final convergence

    def _compute_sdxl_guidance(self, obj: str, env: str) -> float:
        """Simulate SDXL guidance scale (CFG) based on scene complexity."""
        base = 7.5
        if obj != "main 3D object":
            base += 2.0
        if env != "simple digital environment":
            base += 1.5
        return round(min(base + random.uniform(-0.5, 0.5), 12.0), 1)

    def _compute_confidence(self, obj: str, env: str, tsm: float) -> float:
        score = 0.3
        if obj != "main 3D object":
            score += 0.35
        if env != "simple digital environment":
            score += 0.2
        score += tsm * 0.15
        return round(min(score, 1.0), 2)

    def _render(self, obj, env, palette, tsm, sdxl) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.patch.set_facecolor(palette["bg"])

        # ── Left: 3D scene ───────────────────────────────────────────────
        ax = axes[0]
        ax.set_facecolor(palette["bg"])
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title("3D Scene Output", color=palette["accent"], fontsize=13, fontweight="bold")

        # Environment
        ax.add_patch(plt.Rectangle((0,0), 10, 1.5, color=palette["ground"], zorder=1))
        if "city" in env:
            for x in [0.5, 2, 3.5, 6, 7.5, 8.8]:
                h = random.uniform(2.5, 7)
                ax.add_patch(plt.Rectangle((x,1.5), 1, h, color=palette["accent"], alpha=0.25, zorder=2))
        elif "mountain" in env:
            ax.fill_between([0,2.5,5,7.5,10],[1.5,7,3.5,8,1.5],1.5,color=palette["accent"],alpha=0.25,zorder=2)
        elif "forest" in env:
            for x in np.arange(0.5,10,1.3):
                ax.plot([x,x],[1.5,1.5+random.uniform(2,4)],color="#8B5E3C",lw=3,zorder=2)
                ax.scatter([x],[1.5+random.uniform(2,4)+0.8],s=400,color=palette["accent"],alpha=0.4,zorder=3)
        elif "space" in env:
            xs,ys = np.random.uniform(0,10,100), np.random.uniform(0,10,100)
            ax.scatter(xs,ys,s=np.random.uniform(3,20,100),color="white",alpha=0.6,zorder=1)

        # Object
        ax.add_patch(plt.Circle((5,5), 1.8, color=palette["accent"], alpha=0.08, zorder=4))
        ax.add_patch(plt.Circle((5,5), 1.2, color=palette["accent"], alpha=0.2,  zorder=5))
        ax.add_patch(plt.Circle((5,5), 0.8, color=palette["accent"], alpha=0.4,  zorder=6))
        ax.text(5,5, obj.upper(), ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=7)

        # ── Right: TSM trajectory ────────────────────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor(palette["bg"])
        ax2.set_title("TSM Trajectory (Score Distillation)", color=palette["accent"],
                      fontsize=13, fontweight="bold")
        T = 20
        steps = list(range(T))
        scores = [max(0, min(1, 1-(1/(1+random.uniform(2,8)*np.exp(-0.3*t)))+random.gauss(0,0.04))) for t in steps]
        ax2.plot(steps, scores, color=palette["accent"], linewidth=2.5, zorder=3)
        ax2.fill_between(steps, scores, alpha=0.15, color=palette["accent"])
        ax2.axhline(y=tsm, color="white", linestyle="--", alpha=0.5, linewidth=1.2)
        ax2.text(T-1, tsm+0.03, f"TSM={tsm:.2f}", color="white", fontsize=9, ha="right")
        ax2.set_xlabel("Diffusion Timestep", color="white", fontsize=10)
        ax2.set_ylabel("Score", color="white", fontsize=10)
        ax2.tick_params(colors="white")
        for s in ax2.spines.values(): s.set_color(palette["accent"]); s.set_alpha(0.3)
        ax2.text(0.05, 0.05, f"SDXL Guidance (CFG): {sdxl}",
                 transform=ax2.transAxes, color="white", fontsize=9, alpha=0.7)

        fig.suptitle(f"AGENT 1 — Dreamer XL  ·  {obj.title()}",
                     color=palette["accent"], fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DREAMER_XL, dpi=RENDER_DPI, facecolor=palette["bg"])
        plt.close()
        return OUTPUT_DREAMER_XL

    def _print_output(self, r: dict) -> None:
        print(f"\n    Model        : {r['agent']}")
        print(f"    Method       : {r['method']}")
        print(f"    Object       : {r['object']}")
        print(f"    Environment  : {r['environment']}")
        print(f"    TSM Score    : {r['tsm_score']}")
        print(f"    SDXL CFG     : {r['sdxl_guidance']}")
        print(f"    Confidence   : {r['confidence']:.0%}")
        print(f"    Output       : {r['output']}")
