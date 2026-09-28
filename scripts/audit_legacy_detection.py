#!/usr/bin/env python3
"""Reconstruct legacy hysteresis/filter attrition from its saved probability map."""
from pathlib import Path
import json
import argparse
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'reports/legacy_reference/Plate1_B2_14_masks.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/legacy_detection_audit.json')
    args=parser.parse_args()
    with np.load(args.input) as arrays:
        nuclei=arrays['nuclear_mask']; stored=arrays['blob_mask']; probability=arrays['blob_prob_map']
    areas=np.bincount(nuclei.ravel())
    low,high=1/(1+np.exp(1)),1/(1+np.exp(-1))
    raw=apply_hysteresis_threshold(probability,low=low,high=high)
    components,count=ndi.label(raw,structure=np.ones((3,3),bool))
    base=len(areas)
    codes=components[raw].astype(np.int64)*base+nuclei[raw]
    pairs,counts=np.unique(codes,return_counts=True)
    ids=pairs%base
    fractions=counts/areas[ids]
    accepted=(ids>0)&(counts>=8)&(fractions<=.95)
    oversized=(ids>0)&(counts>=8)&(fractions>.95)
    accepted_ids=np.unique(ids[accepted]); rejected_ids=np.unique(ids[oversized])
    raw_nuclear_counts=np.bincount(nuclei[raw],minlength=base)
    raw_fraction=np.divide(raw_nuclear_counts,areas,out=np.zeros(base),where=areas>0)
    final_values=np.isin(codes,pairs[accepted])
    mismatch=int(np.count_nonzero(final_values != (stored[raw]>0)))+int(np.count_nonzero(stored[~raw]))
    report={
        'source':str(args.input),'nuclei':int(np.count_nonzero(areas[1:])),
        'saved_components':int(stored.max()),'saved_nuclei_with_components':int(len(np.unique(nuclei[stored>0]))),
        'source_saved_run_configuration_available':False,
        'reconstruction_assumed_config':{'hysteresis_high':4.,'hysteresis_low':2.,'min_blob_area':8,'max_blob_fraction':.95},
        'probability_thresholds':{'low':low,'high':high},
        'raw_global_connected_components':int(count),'raw_components_within_nuclei':int((ids>0).sum()),
        'nuclei_with_raw_candidates':int(np.count_nonzero(raw_nuclear_counts[1:])),
        'reconstructed_accepted_components':int(accepted.sum()),'reconstructed_nuclei_with_components':len(accepted_ids),
        'rejected_oversized_components':int(oversized.sum()),'nuclei_with_oversized_components':len(rejected_ids),
        'nuclei_with_only_oversized_rejections':len(set(rejected_ids)-set(accepted_ids)),
        'nuclei_with_accepted_and_oversized_components':len(set(rejected_ids)&set(accepted_ids)),
        'nuclei_total_raw_coverage_gt95percent':int(((raw_fraction[1:]>.95)&(areas[1:]>0)).sum()),
        'nuclei_without_saved_components':int(np.count_nonzero(areas[1:]))-len(accepted_ids),
        'reconstructed_saved_binary_mask_mismatched_pixels':mismatch,
        'caveat':'Saved output has no run configuration. Current archived defaults are tested by exact final-mask reproduction. A monotone sigmoid converts high/low 4/2 to the stated thresholds; saturation at 0 or 1 does not affect those interior thresholds. Occupancy sums per nucleus and oversized per-component rejection are reported separately.',
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
