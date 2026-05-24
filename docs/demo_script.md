# Demo Script — Dreamer World Family: 3-Agent Pipeline

## 🎯 Introduction

Good morning/afternoon, everyone.

Today I'll present a **simplified but functional Python implementation**  
of the Dreamer World Family, using a 3-agent pipeline based on three real papers:

- **Agent 1 → Dreamer XL** (Miao et al., 2024) — Text-to-3D via TSM + SDXL
- **Agent 2 → X-Dreamer** (Ma et al., 2024) — 2D-to-3D domain bridging via CG-LoRA + AMA Loss
- **Agent 3 → ATD** (Zhang et al., 2025) — Language imagination for robot navigation

The goal is not to reproduce full GPU-based diffusion pipelines.  
Instead, this project **demonstrates the same logical and mathematical pipeline**  
in pure Python, runnable on any machine.

---

## 🔁 The 3-Agent Pipeline

```
Text Prompt
     │
     ▼
┌────────────────────────────────────────┐
│  Agent 1 — Dreamer XL                 │
│  Text → 3D via TSM + SDXL             │
│  Output: dreamer_xl_scene.png          │
└────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────┐
│  Agent 2 — X-Dreamer                  │
│  2D → 3D via CG-LoRA + AMA Loss       │
│  Output: xdreamer_scene.png            │
└────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────┐
│  Agent 3 — ATD                        │
│  Language → Navigation (Waypoints)    │
│  Output: atd_navmap.png               │
└────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────┐
│  Final Comparison                     │
│  Radar + Ranking + Summary            │
│  Output: comparison.png               │
└────────────────────────────────────────┘
     │
     ▼
  scene_report.json  (full structured export)
```

---

## 🤖 Agent 1 — Dreamer XL

**Paper:** *Dreamer XL: Towards High-Resolution Text-to-3D Generation via Trajectory Score Matching* (Miao et al., 2024)

**What it simulates:**

1. **Trajectory Score Matching (TSM):**  
   Runs T=20 diffusion timesteps. At each step, a score is computed simulating  
   the convergence of the 3D object along the diffusion trajectory.  
   Final TSM score = mean of the last 5 steps (convergence zone).

2. **SDXL Guidance (CFG scale):**  
   The Classifier-Free Guidance scale is computed based on scene complexity  
   (object detected + environment type). Ranges from 7.5 to 12.0.

3. **Rendering:**  
   Left panel: 3D-styled scene with environment (city, mountain, forest, space…).  
   Right panel: TSM convergence trajectory plot.

**Demo output:**
```
    Object       : dragon
    Environment  : mountain landscape
    TSM Score    : 0.812
    SDXL CFG     : 11.0
    Confidence   : 85%
```

**Link to real paper:**  
In Dreamer XL, TSM ensures multi-view consistency of the 3D object along  
a trajectory of diffusion timesteps, guided by SDXL as a 2D visual prior.

---

## 🌍 Agent 2 — X-Dreamer

**Paper:** *X-Dreamer: Creating High-quality 3D Content by Bridging the Domain Gap Between Text-to-2D and Text-to-3D Generation* (Ma et al., 2024)

**What it simulates:**

1. **CG-LoRA (Camera-Guided Low-Rank Adaptation):**  
   Selects a LoRA rank ∈ {4, 8, 16, 32, 64} based on prompt complexity.  
   Higher rank = richer camera viewpoint adaptation.

2. **AMA Loss (Attention Map Alignment Loss):**  
   Simulates 50 training iterations of the alignment loss converging  
   toward zero, reducing the domain gap between 2D diffusion and 3D geometry.

3. **Rendering:**  
   Top: 6-view multi-camera grid (front, side, back, top, isometric, low angle)  
   with simulated 2D attention maps (gaussian blobs per viewpoint).  
   Bottom: AMA loss curve + CG-LoRA rank bar + summary panel.

**Demo output:**
```
    CG-LoRA Rank : 8
    AMA Loss     : 0.0312
    Domain Gap   : 0.0298
    Confidence   : 90%
```

**Link to real paper:**  
In X-Dreamer, CG-LoRA injects camera embeddings into the UNet attention layers  
via low-rank matrices. AMA Loss forces 2D attention maps to align with 3D geometry,  
eliminating the domain gap that degrades text-to-3D quality.

---

## 🧭 Agent 3 — ATD (Adaptive Text Dreamer)

**Paper:** *Adaptive Text Dreamer: Zero-Shot Robot Navigation via Language Imagination* (Zhang et al., 2025)

**What it simulates:**

1. **Language instruction extraction:**  
   Detects navigation keywords (go, find, explore, reach…) from the text prompt  
   and maps them to structured navigation instructions.

2. **Waypoint generation:**  
   Generates N imagined waypoints on a 10×10 navigation grid,  
   seeded from the prompt content for reproducibility.

3. **Language grounding scores:**  
   Assigns a grounding score per waypoint measuring how well the text instruction  
   aligns with the imagined spatial position (0.0 = no alignment, 1.0 = perfect).

4. **Path efficiency:**  
   Ratio of straight-line distance to actual path length (1.0 = optimal straight path).

5. **Rendering:**  
   Left: Top-down navigation map with imagined path, waypoints, obstacles, grounding scores.  
   Right: Per-waypoint grounding bar chart + instructions + path efficiency.

**Demo output:**
```
    Target          : dragon
    Waypoints       : 6
    Avg Grounding   : 0.723
    Path Efficiency : 0.541
    Confidence      : 78%
```

**Link to real paper:**  
In ATD, the robot imagines a full traversal of the environment from the text instruction  
before physically navigating, enabling zero-shot navigation in unseen environments.

---

## 📊 Final Comparison

The `comparison.py` module aggregates all 3 results and generates a 6-panel figure:

| Panel | Content |
|---|---|
| Top-left | Confidence bar chart (%) |
| Top-center | Multi-criteria radar chart |
| Top-right | Agent methods summary |
| Bottom-left | Per-metric grouped bar chart |
| Bottom-center | Final ranking with medals |
| Bottom-right | Pipeline summary text |

**Metrics per agent:**

| Agent | Metric 1 | Metric 2 | Metric 3 |
|---|---|---|---|
| Dreamer XL | TSM score | SDXL CFG (normalized) | Object detection |
| X-Dreamer | 1 − AMA loss | 1 − Domain gap | LoRA rank (normalized) |
| ATD | Avg grounding | Path efficiency | Waypoints (normalized) |

---

## 🔗 Mapping to Real Dreamer Systems

| This Project | Real Dreamer World Family |
|---|---|
| TSM simulation (T=20 steps) | TSM optimization over diffusion timesteps |
| SDXL CFG heuristic | Score Distillation Sampling from SDXL |
| CG-LoRA rank selection | Low-rank camera embedding injection in UNet |
| AMA loss curve (50 iters) | Attention Map Alignment Loss training |
| Waypoint imagination from text | Language-conditioned navigation imagination |
| Matplotlib rendering | NeRF / 3D Gaussian Splatting |
| Heuristic confidence | SDS quality / CLIP alignment score |

---

## ✅ Conclusion

This project demonstrates that the **core logic of the Dreamer World Family**  
can be understood, implemented, and visualized in pure Python:

```
Text Prompt
  → Agent 1 (Dreamer XL)  : simulates TSM + SDXL guidance for Text→3D
  → Agent 2 (X-Dreamer)   : simulates CG-LoRA + AMA Loss for 2D→3D bridging
  → Agent 3 (ATD)         : simulates language imagination for robot navigation
  → Comparison            : multi-criteria ranking of all 3 agents
```

It serves as a **pedagogical bridge** between the theoretical papers  
and the actual engineering of state-of-the-art text-to-3D generation systems.

Thank you.
