#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the final tanker stable-head C4 reverse/zero audit used in revision.

Input
-----
Table_C4_02_primary10_row_level_joint_support.csv produced by
run_tanker_C4_harmonization_AIStudio_OVERNIGHT_AUDITED_REGENERATED.py.

Outputs
-------
Table_C4_08_HEADSEA_reverse_zero_audit.csv
Table_C4_09_HEADSEA_reverse_cutoff_sensitivity.csv
Table_C4_10_HEADSEA_by_vessel_direction.csv

The primary sign rule is epsilon=0, matching the original C4 implementation.
Additional epsilon values are sensitivity checks for near-zero responses only.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--root',
        default='tanker_C4_harmonization_overnight',
        help='Root output directory of the tanker harmonisation runner.',
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    c4dir = root / '03_wave_c4'
    src = c4dir / 'Table_C4_02_primary10_row_level_joint_support.csv'
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src)
    required = {
        'vessel_id','wave_perturbation_pct','sector_original','sector_harmonized',
        'delta_original','delta_harmonized','sector_stable'
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    df = df[np.isclose(pd.to_numeric(df['wave_perturbation_pct'], errors='coerce'), 10.0)].copy()
    stable = df['sector_stable']
    if stable.dtype != bool:
        stable = stable.astype(str).str.strip().str.lower().isin(['true','1','yes'])

    g = df[
        stable
        & df['sector_original'].astype(str).str.lower().eq('head')
        & df['sector_harmonized'].astype(str).str.lower().eq('head')
    ].copy()

    if len(g) != 8627:
        raise AssertionError(f'Expected 8,627 primary stable-head rows, got {len(g):,}')
    if g['vessel_id'].astype(str).nunique() != 10:
        raise AssertionError('Expected all 10 tanker vessels in primary stable-head subgroup')

    eps_list = [0.0, 1e-6, 1e-5, 1e-4]
    branches = {'original':'delta_original', 'harmonized_ERA5':'delta_harmonized'}
    rows, vessel_rows = [], []

    for branch, col in branches.items():
        x_all = pd.to_numeric(g[col], errors='coerce')
        if not x_all.notna().all():
            raise ValueError(f'{branch}: NaN in delta')

        for vessel, vg in g.groupby(g['vessel_id'].astype(str)):
            xv = pd.to_numeric(vg[col], errors='coerce')
            for eps in eps_list:
                if eps == 0:
                    expected, reverse, nearzero = xv > 0, xv < 0, xv == 0
                else:
                    expected, reverse, nearzero = xv > eps, xv < -eps, xv.abs() <= eps
                vessel_rows.append({
                    'branch': branch, 'vessel_id': vessel, 'epsilon_t_per_10min': eps,
                    'n': len(xv), 'expected_n': int(expected.sum()),
                    'reverse_n': int(reverse.sum()), 'nearzero_n': int(nearzero.sum()),
                    'expected_pct': 100*expected.mean(), 'reverse_pct': 100*reverse.mean(),
                    'nearzero_pct': 100*nearzero.mean(), 'mean_delta': xv.mean(),
                    'median_delta': xv.median(),
                })

        for eps in eps_list:
            if eps == 0:
                expected, reverse, nearzero = x_all > 0, x_all < 0, x_all == 0
            else:
                expected, reverse, nearzero = x_all > eps, x_all < -eps, x_all.abs() <= eps

            by_vessel=[]
            for _, vg in g.groupby(g['vessel_id'].astype(str)):
                xv = pd.to_numeric(vg[col], errors='coerce')
                if eps == 0:
                    ev, rv, zv = (xv > 0).mean(), (xv < 0).mean(), (xv == 0).mean()
                else:
                    ev, rv, zv = (xv > eps).mean(), (xv < -eps).mean(), (xv.abs() <= eps).mean()
                by_vessel.append((ev,rv,zv))
            by_vessel=np.asarray(by_vessel,float)

            rows.append({
                'branch': branch, 'epsilon_t_per_10min': eps, 'n': len(x_all),
                'n_vessels': g['vessel_id'].astype(str).nunique(),
                'pooled_expected_n': int(expected.sum()), 'pooled_reverse_n': int(reverse.sum()),
                'pooled_nearzero_n': int(nearzero.sum()),
                'pooled_expected_pct': 100*expected.mean(), 'pooled_reverse_pct': 100*reverse.mean(),
                'pooled_nearzero_pct': 100*nearzero.mean(),
                'vessel_balanced_expected_pct': 100*by_vessel[:,0].mean(),
                'vessel_balanced_reverse_pct': 100*by_vessel[:,1].mean(),
                'vessel_balanced_nearzero_pct': 100*by_vessel[:,2].mean(),
                'mean_delta_t_per_10min': x_all.mean(),
                'median_delta_t_per_10min': x_all.median(),
            })

    audit = pd.DataFrame(rows)
    by_vessel = pd.DataFrame(vessel_rows)
    primary = audit[np.isclose(audit['epsilon_t_per_10min'],0.0)].copy()

    cutoff_rows=[]
    for _,r in primary.iterrows():
        for cutoff in [25.0,35.0,45.0]:
            cutoff_rows.append({
                'branch': r['branch'], 'reverse_cutoff_pct': cutoff,
                'pooled_reverse_pct': r['pooled_reverse_pct'],
                'vessel_balanced_reverse_pct': r['vessel_balanced_reverse_pct'],
                'pooled_C4_reverse_threshold_triggered': bool(r['pooled_reverse_pct'] > cutoff),
                'vessel_balanced_C4_reverse_threshold_triggered': bool(r['vessel_balanced_reverse_pct'] > cutoff),
            })
    cutoff = pd.DataFrame(cutoff_rows)

    out1=c4dir/'Table_C4_08_HEADSEA_reverse_zero_audit.csv'
    out2=c4dir/'Table_C4_09_HEADSEA_reverse_cutoff_sensitivity.csv'
    out3=c4dir/'Table_C4_10_HEADSEA_by_vessel_direction.csv'
    audit.to_csv(out1,index=False,encoding='utf-8-sig')
    cutoff.to_csv(out2,index=False,encoding='utf-8-sig')
    by_vessel.to_csv(out3,index=False,encoding='utf-8-sig')

    print(primary.to_string(index=False))
    print('\nReverse-cutoff sensitivity')
    print(cutoff.to_string(index=False))
    print('\nSaved:')
    for p in (out1,out2,out3): print(p)

if __name__ == '__main__':
    main()
