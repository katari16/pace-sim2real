#!/usr/bin/env python3
"""
Fix the time vector offset in chirp_data.pt

The GO2 data collection script incorrectly set start_time at script start
instead of when Phase 4 (chirp playback) begins. This results in a ~3 second
offset in the time vector.

This script normalizes the time vector to start at t=0.
"""

import torch
from pathlib import Path

def fix_time_offset(input_path: str, output_path: str = None):
    """
    Normalize the time vector in chirp_data.pt to start at t=0.

    Args:
        input_path: Path to the original chirp_data.pt file
        output_path: Path to save the fixed file (defaults to overwriting input)
    """
    if output_path is None:
        output_path = input_path

    # Load the data
    data = torch.load(input_path)

    print("=== BEFORE FIX ===")
    print(f"Time range: {data['time'][0].item():.4f}s to {data['time'][-1].item():.4f}s")
    print(f"Duration: {data['time'][-1].item() - data['time'][0].item():.4f}s")
    print(f"Timesteps: {len(data['time'])}")

    # Normalize time to start at 0
    t0 = data['time'][0].clone()
    data['time'] = data['time'] - t0

    print("\n=== AFTER FIX ===")
    print(f"Time range: {data['time'][0].item():.4f}s to {data['time'][-1].item():.4f}s")
    print(f"Duration: {data['time'][-1].item() - data['time'][0].item():.4f}s")
    print(f"Timesteps: {len(data['time'])}")

    # Save the fixed data
    torch.save(data, output_path)
    print(f"\nFixed data saved to: {output_path}")

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix time offset in chirp_data.pt")
    parser.add_argument("--input", type=str, default="data/go2_sim/chirp_data.pt",
                        help="Input chirp_data.pt file path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: overwrite input)")
    parser.add_argument("--backup", action="store_true",
                        help="Create a backup before overwriting")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)

    # Create backup if requested
    if args.backup and args.output is None:
        backup_path = input_path.with_suffix('.pt.backup')
        import shutil
        shutil.copy(input_path, backup_path)
        print(f"Backup created: {backup_path}")

    fix_time_offset(str(input_path), args.output)