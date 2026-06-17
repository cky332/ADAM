#!/usr/bin/env python3
"""Run the full ADAM reproduction suite end-to-end.

    python run_all.py

Produces, under results/:
    table1.csv                  (Table 1: 5 attacks x 3 agents x 4 LLMs)
    ablations.csv  fig3_*.png    (Figure 3: 8 ablations)
    oracle.csv     fig4_*.png    (Figure 4: oracle vs. estimation)
    defenses.csv   fig5_*.png    (Figure 5: 4 defenses)
    convergence.csv fig6_*.png   (Figure 6/7 + early-stop + rate control)
"""
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

STEPS = [
    ("Table 1  (main results)", "experiments/run_main_table.py"),
    ("Figure 3 (ablations)", "experiments/run_ablations.py"),
    ("Figure 4 (oracle)", "experiments/run_oracle.py"),
    ("Figure 5 (defenses)", "experiments/run_defenses.py"),
    ("Figure 6 (convergence)", "experiments/run_convergence.py"),
]


def main():
    Path("results").mkdir(exist_ok=True)
    for title, script in STEPS:
        print("\n" + "=" * 70 + f"\n {title}\n" + "=" * 70)
        t0 = time.time()
        runpy.run_path(script, run_name="__main__")
        print(f"  ... done in {time.time() - t0:.1f}s")
    print("\nAll experiments complete. See results/.")


if __name__ == "__main__":
    main()
