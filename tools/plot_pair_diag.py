"""Plot the per-tick diagnostic CSV produced by SO110Follower when
config.diagnostic_csv is set.

One subplot per paired joint, showing:
  - outgoing normalized goal (commanded)
  - primary present position (normalized)
  - secondary present position (normalized)
  - pair disagreement (primary - secondary, normalized)

Usage:
    python tools/plot_pair_diag.py <path/to/pair_diag.csv>
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PAIRED_JOINTS = ["elbow_lift", "shoulder_swing", "shoulder_lift"]


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(len(PAIRED_JOINTS), 2, figsize=(14, 3 * len(PAIRED_JOINTS)), sharex=True)
    for i, joint in enumerate(PAIRED_JOINTS):
        goal_col = f"{joint}_goal"
        p_col = f"{joint}_present"
        s_col = f"{joint}_secondary_present"
        if not all(c in df.columns for c in (goal_col, p_col, s_col)):
            print(f"missing columns for {joint}")
            continue

        ax_pos = axes[i, 0]
        ax_pos.plot(df["t"], df[goal_col], label="goal", color="black", linewidth=1.2)
        ax_pos.plot(df["t"], df[p_col], label="primary present", color="C0", alpha=0.8)
        ax_pos.plot(df["t"], df[s_col], label="secondary present", color="C3", alpha=0.8)
        ax_pos.set_title(f"{joint}: positions (normalized)")
        ax_pos.set_ylabel("pos")
        ax_pos.legend(loc="best", fontsize=8)
        ax_pos.grid(alpha=0.3)

        ax_diff = axes[i, 1]
        diff = df[p_col] - df[s_col]
        err_p = df[goal_col] - df[p_col]
        err_s = df[goal_col] - df[s_col]
        ax_diff.plot(df["t"], diff, label="primary - secondary", color="purple", linewidth=1.2)
        ax_diff.plot(df["t"], err_p, label="goal - primary", color="C0", alpha=0.5)
        ax_diff.plot(df["t"], err_s, label="goal - secondary", color="C3", alpha=0.5)
        ax_diff.axhline(0, color="grey", linewidth=0.5)
        ax_diff.set_title(f"{joint}: pair disagreement + tracking error")
        ax_diff.set_ylabel("delta")
        ax_diff.legend(loc="best", fontsize=8)
        ax_diff.grid(alpha=0.3)

    axes[-1, 0].set_xlabel("t (s)")
    axes[-1, 1].set_xlabel("t (s)")
    fig.tight_layout()

    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
