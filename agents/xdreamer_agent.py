# agents/xdreamer_agent.py — Agent 2: X-Dreamer
# Simulates: 2D → 3D domain gap bridging via CG-LoRA + AMA Loss

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from core.utils import setup_logger
from config import OUTPUT_DIR, OUTPUT_XDREAMER, RENDER_DPI, ENVIRONMENT_PALETTES

logger = setup_logger()


class XDreamerAgent:
    """
    Agent 2 — X-Dreamer Simulation
    ================================
    Real paper: X-Dreamer (Ma et al., 2024)

    Core concepts simulated:
    - CG-LoRA (Camera-Guided Low-Rank Adaptation): adapts the 2D diffusion
      model to understand 3D camera viewpoints by injecting camera embeddings
      into the attention layers via low-rank matrices.
    - AMA Loss (Attention Map Alignment Loss): aligns the 2D attention maps
      with the expected 3D geometry, reducing the domain gap between
      2D diffusion priors and 3D object generation.

    This agent:
      1. Takes the text prompt and simulates multi-view camera sampling.
      2. Computes a CG-LoRA rank and AMA loss curve.
      3. Renders a multi-view grid showing the 2D→3D bridging process.
      4. Outputs: xdreamer_scene.png + confidence score.
    """

    OBJECT_KEYWORDS = {
        "robot": "robot", "dragon": "dragon", "castle": "castle",
        "car": "car", "spaceship": "spaceship", "warrior": "warrior",
        "crystal": "crystal", "wolf": "wolf", "volcano": "volcano",
        "submarine": "submarine", "panda": "panda", "tree": "tree",
    }

    CAMERA_VIEWS = [
        "Front View (0°)", "Side View (90°)", "Back View (180°)",
        "Top View (270°)", "Isometric (45°)", "Low Angle (315°)",
    ]

    def run(self, prompt: str) -> dict:
        """Run the X-Dreamer 2D→3D bridging pipeline."""
        logger.info("XDreamer — starting 2D→3D domain bridging...")
        lower = prompt.lower()

        main_object  = self._detect_object(lower)
        lora_rank    = self._compute_lora_rank(prompt)
        ama_loss     = self._simulate_ama_loss(lora_rank)
        domain_gap   = self._compute_domain_gap(ama_loss)
        confidence   = self._compute_confidence(main_object, lora_rank, ama_loss)

        palette = ENVIRONMENT_PALETTES.get("outer space", ENVIRONMENT_PALETTES["simple digital environment"])
        # X-Dreamer uses a neutral dark palette to emphasize multi-view
        palette = {"bg": "#0d0d1a", "accent": "#ff6b6b", "ground": "#1a0a0a"}

        output_path = self._render(main_object, lora_rank, ama_loss, domain_gap, palette)

        result = {
            "agent":       "X-Dreamer",
            "method":      "CG-LoRA + AMA Loss",
            "prompt":      prompt,
            "object":      main_object,
            "lora_rank":   lora_rank,
            "ama_loss":    round(ama_loss[-1], 4),
            "domain_gap":  domain_gap,
            "confidence":  confidence,
            "output":      output_path,
        }

        logger.info(f"XDreamer — done. Confidence: {confidence:.0%} | Domain gap: {domain_gap:.3f}")
        self._print_output(result)
        return result

    # ── Private ─────────────────────────────────────────────────────────────

    def _detect_object(self, prompt: str) -> str:
        for k, v in self.OBJECT_KEYWORDS.items():
            if k in prompt:
                return v
        return "3D object"

    def _compute_lora_rank(self, prompt: str) -> int:
        """
        CG-LoRA rank: higher complexity prompts need higher rank adapters.
        Rank is typically in {4, 8, 16, 32, 64}.
        """
        complexity = len(prompt.split())
        if complexity <= 5:  return 4
        if complexity <= 10: return 8
        if complexity <= 20: return 16
        if complexity <= 35: return 32
        return 64

    def _simulate_ama_loss(self, lora_rank: int) -> list:
        """
        Simulate AMA Loss curve over training iterations.
        Higher LoRA rank → faster convergence (lower final loss).
        """
        iters = 50
        convergence_rate = 0.1 + (lora_rank / 64) * 0.15
        losses = []
        current = 1.0
        for i in range(iters):
            current = current * (1 - convergence_rate) + random.gauss(0, 0.01)
            losses.append(max(0.01, current))
        return losses

    def _compute_domain_gap(self, ama_loss: list) -> float:
        """Domain gap is proportional to the final AMA loss."""
        return round(float(np.mean(ama_loss[-5:])), 4)

    def _compute_confidence(self, obj: str, rank: int, ama_loss: list) -> float:
        score = 0.3
        if obj != "3D object":
            score += 0.3
        score += (rank / 64) * 0.2
        score += max(0, (1 - ama_loss[-1])) * 0.2
        return round(min(score, 1.0), 2)

    def _render(self, obj, rank, ama_loss, domain_gap, palette) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig = plt.figure(figsize=(14, 7))
        fig.patch.set_facecolor(palette["bg"])

        # ── Top row: multi-view grid (6 camera views) ────────────────────
        for i, view_name in enumerate(self.CAMERA_VIEWS):
            ax = fig.add_subplot(2, 6, i + 1)
            ax.set_facecolor(palette["bg"])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(palette["accent"]); s.set_alpha(0.4)

            # Simulate 2D attention map (gaussian blob)
            angle = i * (360 / 6)
            cx = 0.5 + 0.15 * np.cos(np.radians(angle))
            cy = 0.5 + 0.15 * np.sin(np.radians(angle))
            x = np.linspace(0, 1, 30); y = np.linspace(0, 1, 30)
            X, Y = np.meshgrid(x, y)
            Z = np.exp(-((X - cx)**2 + (Y - cy)**2) / 0.05)
            ax.contourf(X, Y, Z, levels=8, cmap="RdPu", alpha=0.8)
            ax.text(0.5, 0.5, obj.upper()[:6], ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white",
                    transform=ax.transAxes)
            ax.set_title(view_name, fontsize=7, color=palette["accent"], pad=3)

        # ── Bottom left: AMA Loss curve ───────────────────────────────────
        ax_loss = fig.add_subplot(2, 3, 4)
        ax_loss.set_facecolor(palette["bg"])
        iters = list(range(len(ama_loss)))
        ax_loss.plot(iters, ama_loss, color=palette["accent"], linewidth=2)
        ax_loss.fill_between(iters, ama_loss, alpha=0.2, color=palette["accent"])
        ax_loss.axhline(y=domain_gap, color="white", linestyle="--", alpha=0.5, lw=1.2)
        ax_loss.text(len(iters)-1, domain_gap+0.01, f"Gap={domain_gap:.3f}",
                     color="white", fontsize=8, ha="right")
        ax_loss.set_title("AMA Loss (Attention Map Alignment)", color=palette["accent"], fontsize=10)
        ax_loss.set_xlabel("Training Iterations", color="white", fontsize=9)
        ax_loss.set_ylabel("Loss", color="white", fontsize=9)
        ax_loss.tick_params(colors="white")
        for s in ax_loss.spines.values(): s.set_color(palette["accent"]); s.set_alpha(0.3)

        # ── Bottom center: CG-LoRA rank visualization ─────────────────────
        ax_lora = fig.add_subplot(2, 3, 5)
        ax_lora.set_facecolor(palette["bg"])
        ranks = [4, 8, 16, 32, 64]
        colors = [palette["accent"] if r == rank else "#444466" for r in ranks]
        bars = ax_lora.bar([str(r) for r in ranks], ranks, color=colors, alpha=0.85)
        ax_lora.set_title("CG-LoRA Rank Selected", color=palette["accent"], fontsize=10)
        ax_lora.set_xlabel("LoRA Rank", color="white", fontsize=9)
        ax_lora.set_ylabel("Value", color="white", fontsize=9)
        ax_lora.tick_params(colors="white")
        ax_lora.text(0.5, 0.9, f"Selected: rank={rank}", ha="center",
                     transform=ax_lora.transAxes, color="white", fontsize=9)
        for s in ax_lora.spines.values(): s.set_color(palette["accent"]); s.set_alpha(0.3)

        # ── Bottom right: Domain gap summary ─────────────────────────────
        ax_info = fig.add_subplot(2, 3, 6)
        ax_info.set_facecolor(palette["bg"])
        ax_info.set_xticks([]); ax_info.set_yticks([])
        for s in ax_info.spines.values(): s.set_visible(False)
        info = (
            f"Object     : {obj}\n"
            f"CG-LoRA    : rank={rank}\n"
            f"AMA Loss   : {ama_loss[-1]:.4f}\n"
            f"Domain Gap : {domain_gap:.4f}\n"
            f"Views      : {len(self.CAMERA_VIEWS)}\n"
        )
        ax_info.text(0.1, 0.5, info, transform=ax_info.transAxes,
                     color="white", fontsize=10, va="center", fontfamily="monospace")
        ax_info.set_title("X-Dreamer Summary", color=palette["accent"], fontsize=10)

        fig.suptitle(f"AGENT 2 — X-Dreamer  ·  2D→3D Bridging  ·  {obj.title()}",
                     color=palette["accent"], fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_XDREAMER, dpi=RENDER_DPI, facecolor=palette["bg"])
        plt.close()
        return OUTPUT_XDREAMER

    def _print_output(self, r: dict) -> None:
        print(f"\n    Model        : {r['agent']}")
        print(f"    Method       : {r['method']}")
        print(f"    Object       : {r['object']}")
        print(f"    CG-LoRA Rank : {r['lora_rank']}")
        print(f"    AMA Loss     : {r['ama_loss']}")
        print(f"    Domain Gap   : {r['domain_gap']}")
        print(f"    Confidence   : {r['confidence']:.0%}")
        print(f"    Output       : {r['output']}")
