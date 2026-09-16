from __future__ import annotations

import os
import sys
import argparse
import glob
import pandas as pd


TARGET_COLUMNS = [
    'rms', 'kurtosis', 'skewness', 'crest_factor', 'shape_factor',
    'spectral_centroid', 'spectral_spread', 'spectral_rolloff', 'psd_peak_freq',
    'envelope_mean', 'envelope_kurtosis', 'vmd_mode_1_energy',
    'vmd_mode_2_energy', 'wavelet_scale_2_energy', 'wavelet_scale_4_energy','domain'
]


def discover_input_csv(results_dir: str) -> str | None:
    """Try to find an input CSV in results/ that contains the most target columns."""
    pattern = os.path.join(results_dir, '*.csv')
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    best_path = None
    best_count = -1
    for path in candidates:
        try:
            # Read header only for speed
            df_head = pd.read_csv(path, nrows=0)
            cols = set(df_head.columns)
            count = sum(1 for c in TARGET_COLUMNS if c in cols)
            if 'label' in cols:
                count += 1
            if count > best_count:
                best_count = count
                best_path = path
        except Exception:
            continue
    return best_path


def extract_features(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)

    wanted = TARGET_COLUMNS + ['label']
    available = [c for c in wanted if c in df.columns]

    if not available:
        raise ValueError(f"None of the requested columns found in {input_csv}. Columns present: {list(df.columns)[:20]} ...")

    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print(f"Warning: missing columns will be skipped: {missing}")

    # Keep order as defined in wanted
    ordered = [c for c in wanted if c in available]
    df_out = df[ordered].copy()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df_out)} rows, {len(df_out.columns)} columns to {output_csv}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Extract selected final features to results/final_features.csv')
    parser.add_argument('--input', '-i', default=None, help='Path to input CSV (default: results/bearing_features.csv or best match in results/)')
    parser.add_argument('--output', '-o', default=os.path.join('results', 'final_features.csv'), help='Path to output CSV')
    args = parser.parse_args(argv)

    # Determine input path
    input_path = args.input
    default_candidate = os.path.join('results', 'comprehensive_features.csv')

    if input_path is None:
        if os.path.isfile(default_candidate):
            input_path = default_candidate
        else:
            discovered = discover_input_csv('results')
            if discovered is None:
                print("Error: No input CSV provided and no CSV found in results/.")
                print("Provide --input path to a CSV that contains the requested columns.")
                return 2
            input_path = discovered

    if not os.path.isfile(input_path):
        print(f"Error: input CSV not found: {input_path}")
        return 2

    try:
        extract_features(input_path, args.output)
    except Exception as e:
        print(f"Failed to extract features: {e}")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


