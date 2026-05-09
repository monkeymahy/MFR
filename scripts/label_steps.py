#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label STEP files with Hole / Boss / Chamfer features.

Usage:
    python scripts/label_steps.py <steps_dir> [labels_dir]

    steps_dir : folder containing .step / .stp files
    labels_dir: output folder for JSON label files (default: <steps_dir>/../labels)

Output format (one JSON per STEP file):
{
    "source": "example.step",
    "features": [
        {
            "type": "Hole",
            "subtype": "Through",
            "face_indices": [3, 5],
            "axis": [0.0, 0.0, 1.0],
            "radius": 5.0,
            "depth": 0.0,
            "center": [10.0, 20.0, 0.0]
        },
        ...
    ],
    "summary": {
        "total_features": 3,
        "hole_count": 1,
        "boss_count": 1,
        "chamfer_count": 1
    }
}
"""

import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def process_single(step_path, label_path, verbose=False):
    """Process one STEP file and write the label JSON."""
    from mfr.feature_recognizer import recognize_features_from_step, features_to_json

    basename = os.path.basename(step_path)

    if verbose:
        print(f"  Processing {basename} ...", end=" ", flush=True)

    t0 = time.time()
    try:
        features = recognize_features_from_step(step_path)
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    result = features_to_json(features)
    result["source"] = basename

    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(label_path) or ".", exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if verbose:
        n = len(features)
        s = result["summary"]
        print(f"{n} feature(s) "
              f"(H:{s['hole_count']} B:{s['boss_count']} C:{s['chamfer_count']}) "
              f"[{elapsed:.2f}s]")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Label STEP files with Hole/Boss/Chamfer features"
    )
    parser.add_argument(
        "steps_dir",
        help="Directory containing .step / .stp files",
    )
    parser.add_argument(
        "labels_dir",
        nargs="?",
        default=None,
        help="Output directory for JSON label files (default: steps_dir/../labels)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-file output",
    )
    args = parser.parse_args()

    steps_dir = os.path.abspath(args.steps_dir)
    if not os.path.isdir(steps_dir):
        print(f"Error: {steps_dir} is not a directory")
        sys.exit(1)

    if args.labels_dir:
        labels_dir = os.path.abspath(args.labels_dir)
    else:
        labels_dir = os.path.join(os.path.dirname(steps_dir), "labels")

    # Collect STEP files
    step_files = sorted(
        f for f in os.listdir(steps_dir)
        if f.lower().endswith((".step", ".stp"))
    )

    if not step_files:
        print(f"No STEP files found in {steps_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Feature Labeling: Hole / Boss / Chamfer")
    print("=" * 60)
    print(f"  Input:   {steps_dir}")
    print(f"  Output:  {labels_dir}")
    print(f"  Files:   {len(step_files)}")
    print()

    os.makedirs(labels_dir, exist_ok=True)

    total_ok = 0
    total_err = 0
    t_start = time.time()

    for i, step_file in enumerate(step_files, 1):
        step_path = os.path.join(steps_dir, step_file)
        model_id = os.path.splitext(step_file)[0]
        label_path = os.path.join(labels_dir, f"{model_id}.json")

        ok = process_single(step_path, label_path, verbose=not args.quiet)
        if ok:
            total_ok += 1
        else:
            total_err += 1

        # Progress
        if not args.quiet and i % max(1, len(step_files) // 20) == 0:
            pct = i / len(step_files) * 100
            print(f"  Progress: {i}/{len(step_files)} ({pct:.0f}%)")

    elapsed = time.time() - t_start

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Succeeded: {total_ok}")
    print(f"  Failed:    {total_err}")
    print(f"  Total:     {len(step_files)}")
    print(f"  Time:      {elapsed:.1f}s")
    if total_ok > 0:
        print(f"  Speed:     {total_ok / elapsed:.1f} files/s")
    print()

    # Aggregate statistics
    if total_ok > 0:
        agg = {"hole": 0, "boss": 0, "chamfer": 0, "total": 0}
        for step_file in step_files:
            model_id = os.path.splitext(step_file)[0]
            label_path = os.path.join(labels_dir, f"{model_id}.json")
            if not os.path.exists(label_path):
                continue
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            s = data.get("summary", {})
            agg["hole"] += s.get("hole_count", 0)
            agg["boss"] += s.get("boss_count", 0)
            agg["chamfer"] += s.get("chamfer_count", 0)
            agg["total"] += s.get("total_features", 0)

        print(f"  Aggregated features across dataset:")
        print(f"    Holes:    {agg['hole']}")
        print(f"    Bosses:   {agg['boss']}")
        print(f"    Chamfers: {agg['chamfer']}")
        print(f"    Total:    {agg['total']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
