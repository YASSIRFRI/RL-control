"""
Script to continue training from a checkpoint with the updated reward function.

Usage:
    python continue_training.py --resume-from models/best_model.pt --total-timesteps 500000

The reward function has been updated to penalize:
1. High joint velocities (reduces shaking)
2. Large torque actions (encourages gentle control)
3. Rapid action changes (encourages smooth control)

These penalties are small (0.005-0.01) to allow smooth transition from existing policy.
"""

import subprocess
import sys

if __name__ == "__main__":
    args = ["python", "train_ppo.py"] + sys.argv[1:]

    print("=" * 60)
    print("Continuing training with updated reward function")
    print("=" * 60)
    print("\nUpdated reward includes:")
    print("  - Velocity penalty: 0.01 * sum(velocities^2)")
    print("  - Action penalty: 0.005 * sum(actions^2)")
    print("  - Smoothness penalty: 0.01 * sum(action_changes^2)")
    print("\nThese small penalties will help reduce shaking while")
    print("allowing the model to adapt from its existing baseline.")
    print("=" * 60)
    print()

    subprocess.run(args)
