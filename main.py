# main.py — Dreamer World Family: 3 Agents Demo

import sys
import json
import os
from agents.dreamer_xl_agent import DreamerXLAgent
from agents.xdreamer_agent   import XDreamerAgent
from agents.atd_agent        import ATDAgent
from agents.comparison       import run_comparison
from core.utils              import print_header, setup_logger
from config                  import OUTPUT_DIR, OUTPUT_REPORT

logger = setup_logger()


def main() -> None:
    print_header("DREAMER WORLD FAMILY — 3 AGENTS DEMO")
    print("  Agent 1 → Dreamer XL    : Text → 3D  (TSM + SDXL)")
    print("  Agent 2 → X-Dreamer     : 2D  → 3D  (CG-LoRA + AMA Loss)")
    print("  Agent 3 → ATD           : Language Imagination → Navigation")
    print()

    try:
        prompt = input("  Enter your text prompt: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n  [Interrupted]")
        sys.exit(0)

    if not prompt:
        print("  [Error] Prompt cannot be empty.")
        sys.exit(1)

    results = []

    # ── Agent 1: Dreamer XL ───────────────────────────────────────────────
    print_header("AGENT 1 — Dreamer XL  (Text → 3D via TSM + SDXL)")
    agent1 = DreamerXLAgent()
    r1 = agent1.run(prompt)
    results.append(r1)

    # ── Agent 2: X-Dreamer ────────────────────────────────────────────────
    print_header("AGENT 2 — X-Dreamer  (2D → 3D via CG-LoRA + AMA Loss)")
    agent2 = XDreamerAgent()
    r2 = agent2.run(prompt)
    results.append(r2)

    # ── Agent 3: ATD ──────────────────────────────────────────────────────
    print_header("AGENT 3 — ATD  (Language Imagination → Robot Navigation)")
    agent3 = ATDAgent()
    r3 = agent3.run(prompt)
    results.append(r3)

    # ── Comparison ────────────────────────────────────────────────────────
    print_header("FINAL COMPARISON — 3 Agents")
    comparison_path = run_comparison(results)
    print(f"\n    Comparison chart : {comparison_path}")

    # ── JSON Report ───────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {"prompt": prompt, "agents": results}
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print_header("PIPELINE COMPLETE")
    ranked = sorted(results, key=lambda r: r["confidence"], reverse=True)
    medals = ["#1", "#2", "#3"]
    print()
    for medal, r in zip(medals, ranked):
        print(f"  {medal}  {r['agent']:<15} — Confidence: {r['confidence']:.0%}")
    print()
    print(f"  Outputs saved in: {OUTPUT_DIR}/")
    print(f"    dreamer_xl_scene.png")
    print(f"    xdreamer_scene.png")
    print(f"    atd_navmap.png")
    print(f"    comparison.png")
    print(f"    scene_report.json")
    print()


if __name__ == "__main__":
    main()
