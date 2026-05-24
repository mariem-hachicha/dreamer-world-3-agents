# 🌍 Dreamer World Family — 3 Agents Demo

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Educational%20Demo-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Course](https://img.shields.io/badge/Course-AI%20Beyond%20Transformers-purple)

> A professional educational implementation of the **Dreamer World Family** concept,  
> simulating a 3-agent pipeline for text-to-3D scene generation and robot navigation.

---

## 📌 Overview

This project is inspired by three key papers from the **Dreamer World Family**:

| Paper | Method | This Agent |
|---|---|---|
| Dreamer XL — Miao et al., 2024 | TSM + Stable Diffusion XL | Agent 1 |
| X-Dreamer — Ma et al., 2024 | CG-LoRA + AMA Loss | Agent 2 |
| Adaptive Text Dreamer — Zhang et al., 2025 | Language Imagination + Navigation | Agent 3 |

Each agent simulates the core mathematical pipeline of its real counterpart  
using only Python, `matplotlib`, and `numpy` — **no GPU or diffusion model required**.

```
Text Prompt ──► [Agent 1: Dreamer XL]       Text → 3D  (TSM + SDXL)
                        │
                        ▼
               [Agent 2: X-Dreamer]         2D  → 3D  (CG-LoRA + AMA Loss)
                        │
                        ▼
               [Agent 3: ATD]               Language  → Navigation (Waypoints)
                        │
                        ▼
               [Final Comparison]           Multi-criteria ranking of the 3 agents
                        │
                        ▼
              outputs/
              ├── dreamer_xl_scene.png
              ├── xdreamer_scene.png
              ├── atd_navmap.png
              ├── comparison.png
              └── scene_report.json
```

---

## 🗂️ Project Structure

```
dreamer_world_pro/
│
├── main.py                        # Entry point — orchestrates the 3 agents
├── config.py                      # Centralized paths, DPI, color palettes
├── requirements.txt               # matplotlib, numpy
├── README.md
│
├── agents/
│   ├── __init__.py
│   ├── dreamer_xl_agent.py        # Agent 1 — TSM + SDXL simulation
│   ├── xdreamer_agent.py          # Agent 2 — CG-LoRA + AMA Loss simulation
│   ├── atd_agent.py               # Agent 3 — Language imagination + navigation
│   └── comparison.py             # Final multi-criteria comparison chart
│
├── core/
│   ├── __init__.py
│   ├── scene.py                   # Scene dataclass (shared data model)
│   └── utils.py                   # Logger & print formatting utilities
│
├── outputs/                       # Auto-generated visual outputs
│   ├── dreamer_xl_scene.png
│   ├── xdreamer_scene.png
│   ├── atd_navmap.png
│   ├── comparison.png
│   └── scene_report.json
│
└── docs/
    └── demo_script.md             # Presentation script for the demo
```

---

## ⚙️ Installation

```bash
# 1. Unzip the project
cd dreamer_world_pro

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the Demo

```bash
python main.py
```

**Example interaction:**

```
Enter your text prompt: A dragon in a mountain forest

══════════════════════════════════════════════════════════
  DREAMER WORLD FAMILY — 3 AGENTS DEMO
══════════════════════════════════════════════════════════
  Agent 1 → Dreamer XL    : Text → 3D  (TSM + SDXL)
  Agent 2 → X-Dreamer     : 2D  → 3D  (CG-LoRA + AMA Loss)
  Agent 3 → ATD           : Language Imagination → Navigation

══════════════════════════════════════════════════════════
  AGENT 1 — Dreamer XL  (Text → 3D via TSM + SDXL)
══════════════════════════════════════════════════════════
    Model        : Dreamer XL
    Method       : Trajectory Score Matching + SDXL
    Object       : dragon
    Environment  : mountain landscape
    TSM Score    : 0.812
    SDXL CFG     : 11.0
    Confidence   : 85%
    Output       : outputs/dreamer_xl_scene.png

══════════════════════════════════════════════════════════
  AGENT 2 — X-Dreamer  (2D → 3D via CG-LoRA + AMA Loss)
══════════════════════════════════════════════════════════
    Model        : X-Dreamer
    Method       : CG-LoRA + AMA Loss
    Object       : dragon
    CG-LoRA Rank : 8
    AMA Loss     : 0.0312
    Domain Gap   : 0.0298
    Confidence   : 90%
    Output       : outputs/xdreamer_scene.png

══════════════════════════════════════════════════════════
  AGENT 3 — ATD  (Language Imagination → Robot Navigation)
══════════════════════════════════════════════════════════
    Model           : ATD
    Method          : Language Imagination + Robot Navigation
    Target          : dragon
    Waypoints       : 6
    Avg Grounding   : 0.723
    Path Efficiency : 0.541
    Confidence      : 78%
    Output          : outputs/atd_navmap.png

══════════════════════════════════════════════════════════
  FINAL COMPARISON — 3 Agents
══════════════════════════════════════════════════════════

  #1  X-Dreamer       — Confidence: 90%
  #2  Dreamer XL      — Confidence: 85%
  #3  ATD             — Confidence: 78%

  Outputs saved in: outputs/
```

---

## 🤖 Agent Descriptions

### Agent 1 — Dreamer XL (`dreamer_xl_agent.py`)

Simulates the **Dreamer XL** pipeline (Miao et al., 2024):

- **Trajectory Score Matching (TSM):** iterates over T=20 diffusion timesteps to compute a convergence score, simulating how TSM minimizes the score gap across a multi-step trajectory.
- **SDXL Guidance (CFG):** computes a Classifier-Free Guidance scale based on scene complexity, simulating the role of Stable Diffusion XL as a 2D visual prior.
- **Renders:** a 3D-styled scene (left) + TSM convergence trajectory (right).

> *In the real paper, TSM optimizes a NeRF/3DGS representation using SDXL as a score distillation prior.*

---

### Agent 2 — X-Dreamer (`xdreamer_agent.py`)

Simulates the **X-Dreamer** pipeline (Ma et al., 2024):

- **CG-LoRA (Camera-Guided Low-Rank Adaptation):** selects a LoRA rank proportional to prompt complexity; higher rank = better camera viewpoint understanding.
- **AMA Loss (Attention Map Alignment Loss):** simulates convergence of the alignment loss over 50 training iterations, reducing the 2D–3D domain gap.
- **Renders:** a 6-view multi-camera grid (top) + AMA loss curve + LoRA rank bar (bottom).

> *In the real paper, CG-LoRA injects camera embeddings into UNet attention layers; AMA Loss aligns 2D attention maps with 3D geometry.*

---

### Agent 3 — ATD (`atd_agent.py`)

Simulates the **Adaptive Text Dreamer** pipeline (Zhang et al., 2025):

- **Language Imagination:** extracts navigation instructions from the text prompt and generates imagined waypoints on a 10×10 navigation grid.
- **Language Grounding:** assigns a grounding score per waypoint measuring alignment between the text instruction and the imagined spatial position.
- **Path Efficiency:** computes straight-line vs. actual path ratio.
- **Renders:** a top-down navigation map with the imagined path (left) + per-waypoint grounding scores (right).

> *In the real paper, ATD enables zero-shot robot navigation in unseen environments using language-conditioned world imagination.*

---

### Comparison (`comparison.py`)

Aggregates results from all 3 agents and generates a 6-panel comparison figure:
- Confidence bar chart
- Multi-criteria radar chart (confidence, quality, alignment, structure)
- Agent methods summary
- Per-metric grouped bar chart
- Final ranking
- Pipeline summary

---

## 🔗 Mapping to Real Dreamer Systems

| This Project | Real Dreamer World Family |
|---|---|
| TSM score simulation (20 steps) | Trajectory Score Matching (TSM) optimization |
| SDXL CFG heuristic | Stable Diffusion XL score distillation (SDS) |
| CG-LoRA rank selection | Low-rank camera embedding injection in UNet |
| AMA loss simulation (50 iters) | Attention Map Alignment Loss training |
| Waypoint generation from text | Language-conditioned imagination for navigation |
| Matplotlib rendering | NeRF / 3D Gaussian Splatting |
| Heuristic confidence score | Score Distillation Sampling (SDS) quality |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `matplotlib` | ≥ 3.7.0 | Scene rendering and visualization |
| `numpy` | ≥ 1.24.0 | Numerical computations |
| `dataclasses` | stdlib | Scene data structure |
| `json` | stdlib | Report export |
| `logging` | stdlib | Runtime logging |

---

## 📚 References

- **Dreamer XL** — Miao et al., 2024. *Dreamer XL: Towards High-Resolution Text-to-3D Generation via Trajectory Score Matching.*
- **X-Dreamer** — Ma et al., 2024. *X-Dreamer: Creating High-quality 3D Content by Bridging the Domain Gap Between Text-to-2D and Text-to-3D Generation.*
- **ATD** — Zhang et al., 2025. *Adaptive Text Dreamer: Zero-Shot Robot Navigation via Language Imagination.*
- **DreamFusion** — Poole et al., 2022. *DreamFusion: Text-to-3D using 2D Diffusion.*

---

## 👥 Authors

> [Your Names Here]  
> Course: AI Beyond Transformers  
> Institution: [Your University]  
> Date: 2026
