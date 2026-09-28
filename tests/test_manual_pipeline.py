"""Integration guarantees for raw, per-cycle, all-nucleus processing."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from popcrop.config import PipelineConfig
from popcrop.manual_pipeline import ManualNucleusAnalyzer
from popcrop.io.codebook import Codebook, BASE_TO_COLOR
from popcrop.annotation.analysis import analyze_nucleus_cycle

class ImageEvidenceCaller:
    def predict_cycle_probabilities(self, frame):
        out=[]
        for _,r in frame.iterrows():
            g,rp = r.green_max > 1000, r.red_max > 500
            base={(False,False):'G',(True,False):'A',(False,True):'T',(True,True):'C'}[g,rp]
            probs={c:0.001 for c in ('red','green','yellow','dark','ambiguous')}
            probs[BASE_TO_COLOR[base]]=0.996
            out.append(probs)
        return out

def analyzer():
    obj=ManualNucleusAnalyzer.__new__(ManualNucleusAnalyzer)
    obj.config=PipelineConfig()
    obj.settings=obj.config.manual_analysis
    obj.base_caller=ImageEvidenceCaller()
    obj.blob_model=None
    return obj

def test_each_cycle_measured_raw_and_negative_masks_are_not_gated():
    shape=(4,2,160,240)
    stack=np.full(shape,100,dtype=np.uint16)
    labels=np.zeros(shape[-2:],dtype=np.int32)
    yy,xx=np.ogrid[:160,:240]
    for label,(y,x) in enumerate([(60,50),(60,130),(60,210)],1):
        labels[(yy-y)**2+(xx-x)**2<20**2]=label
    for t,channels in enumerate([(1,),(0,1),(),(0,)]):
        for ch in channels:stack[t,ch,52:68,42:58]=8000
    # This signal is below the green detector's gate but raw distribution is
    # still passed to the base model. It must never be erased by pre-gating.
    stack[0,1,52:68,202:218]=2000
    a=analyzer()
    cb=Codebook(pd.DataFrame({'barcode':['ACGT','TTTC'],'gene':['one','two']}))
    nuclei,reads,masks=a.run(stack,labels,codebook=cb)
    assert len(nuclei)==3 and len(a.df_cycles)==12
    first=nuclei.iloc[0]
    assert first.called_sequence=='ACGT' and first.decoding_status=='matched'
    assert nuclei.iloc[1].decoding_status=='no_signal'
    assert nuclei.iloc[1].assigned_barcode=='UNASSIGNED'
    assert nuclei.iloc[2].has_signal and not nuclei.iloc[2].has_detectable_spot
    assert 'SIGNAL_WITHOUT_DETECTOR_MASK' in nuclei.iloc[2].qc_flags
    assert len(reads)==2
    row=a.df_cycles.iloc[0]
    ref=analyze_nucleus_cycle(stack[0,1],stack[0,0],None,(60,50))
    assert row.green_median==ref['base_row']['green_median']
    assert json.loads(row.green_mask_pixels)==np.flatnonzero(ref['green_mask']).tolist()
    assert json.loads(row.green_distribution)==ref['base_row']['green_distribution']
    assert stack.dtype==np.uint16 and stack[0,1,60,50]==8000

def test_raw_model_rejects_incompatible_corrections():
    cfg=PipelineConfig()
    cfg.preprocessing.camera_offset=100
    with pytest.raises(ValueError,match='raw intensities'):
        ManualNucleusAnalyzer(cfg)

def test_full_cycle_count_is_validated_before_any_analysis():
    cb=Codebook(pd.DataFrame({'barcode':['AAAA'],'gene':['one']}))
    with pytest.raises(ValueError,match='Image has 2'):
        analyzer().run(np.zeros((2,2,20,20)),np.zeros((20,20),dtype=int),codebook=cb)


def test_remote_nearest_neighbor_ownership_matches_shared_training_contract():
    stack = np.full((1, 2, 140, 680), 100, dtype=np.uint16)
    labels = np.zeros(stack.shape[-2:], dtype=np.int32)
    yy, xx = np.ogrid[:140, :680]
    centers = [(65, 60), (65, 610)]
    for label, (y, x) in enumerate(centers, 1):
        labels[(yy-y)**2+(xx-x)**2 < 15**2] = label
    stack[0, 1, 61:73, 58:70] = 8000
    a = analyzer()
    a.run(stack, labels)
    expected = analyze_nucleus_cycle(stack[0, 1], stack[0, 0], None,
        centers[0], neighbor_centers_yx=[centers[1]])
    assert a.df_cycles.iloc[0].ownership_ratio == expected['base_row']['ownership_ratio']


def test_overlapping_diagnostic_masks_retain_each_nucleus_original_pixels():
    stack = np.full((1, 2, 120, 120), 100, dtype=np.uint16)
    labels = np.zeros(stack.shape[-2:], dtype=np.int32)
    yy, xx = np.ogrid[:120, :120]
    for label, (y, x) in enumerate([(60, 46), (60, 74)], 1):
        labels[(yy-y)**2+(xx-x)**2 < 8**2] = label
    stack[0, :, 55:65, 55:65] = 8000
    a = analyzer()
    nuclei, reads, masks = a.run(stack, labels)
    assert len(nuclei) == len(reads) == len(a.df_cycles) == 2
    recovered = []
    for _, row in a.df_cycles.iterrows():
        provenance = json.loads(row.analysis_provenance)
        box = provenance['crop_box']
        pixels = json.loads(row.green_mask_pixels)
        y, x = np.unravel_index(pixels, (box['height'], box['width']))
        recovered.append(set(zip(y+box['y_min'], x+box['x_min'])))
    assert recovered[0] & recovered[1]
    assert np.count_nonzero(masks['blob_mask']) == len(recovered[0] | recovered[1])


def test_report_counts_accepted_assignments_without_counting_rejected_candidates(tmp_path):
    from popcrop.reporting import write_run_report
    nuclei = pd.DataFrame([
        dict(has_detectable_spot=True, has_signal=True, decoding_status='matched',
             nearest_hamming_distance=0, hamming_distance=0, best_gene='accepted'),
        dict(has_detectable_spot=False, has_signal=True, decoding_status='ambiguous_gene',
             nearest_hamming_distance=1, hamming_distance=1, best_gene='AMBIGUOUS'),
        dict(has_detectable_spot=True, has_signal=False, decoding_status='no_signal',
             nearest_hamming_distance=2, hamming_distance=2, best_gene='rejected'),
    ])
    saved = write_run_report(tmp_path, nuclei, nuclei, PipelineConfig(), {})
    report = json.loads(open(saved['summary_json']).read())
    assert report['total_nuclei'] == 3
    assert report['nuclei_with_detector_spots'] == report['nuclei_with_classified_signal'] == 2
    assert report['accepted_library_assignments'] == 1
    assert report['accepted_barcode_sequences'] == 2
    assert report['unique_assigned_genes'] == 1
    assert report['signal_nuclei_nearest_hamming_histogram'] == {'0': 1, '1': 1}
    genes = pd.read_csv(saved['gene_abundance'])
    assert genes.gene.tolist() == ['accepted']


def test_signal_probability_is_combined_presence_not_largest_individual_class():
    class UncertainPositiveCaller:
        def predict_cycle_probabilities(self, frame):
            return [dict(green=.25, yellow=.24, red=.23, dark=.28, ambiguous=0.) for _ in range(len(frame))]
    a = analyzer()
    a.base_caller = UncertainPositiveCaller()
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[35:65, 35:65] = 1
    nuclei, _, _ = a.run(np.full((1, 2, 100, 100), 100, dtype=np.uint16), labels)
    assert nuclei.iloc[0].called_sequence == 'G'
    assert nuclei.iloc[0].has_signal
    assert nuclei.iloc[0].signal_cycles == 1
    assert 'SIGNAL_WITH_DARK_HARDCALL' in nuclei.iloc[0].qc_flags


def test_cycle_zero_reference_is_used_when_current_dapi_is_blank():
    a = analyzer()
    labels = np.zeros((120, 120), dtype=np.int32)
    labels[50:70, 50:70] = 1
    stack = np.full((1, 2, 120, 120), 100, dtype=np.uint16)
    yy, xx = np.ogrid[:120, :120]
    segmentation_dapi = (100 + 6000*((yy-60)**2+(xx-68)**2 < 15**2)).astype(np.uint16)
    cycle_zero = (100 + 6000*((yy-60)**2+(xx-53)**2 < 15**2)).astype(np.uint16)
    blank_current = np.zeros((1, 120, 120), dtype=np.uint16)
    a.run(stack, labels, dapi_image=segmentation_dapi, dapi_stack=blank_current, reference_dapi=cycle_zero)
    observed = json.loads(a.df_cycles.iloc[0].analysis_provenance)['center_anchored_yx']
    expected = analyze_nucleus_cycle(stack[0, 1], stack[0, 0], blank_current[0], (59.5, 59.5), reference_dapi=cycle_zero)
    assert observed == expected['provenance']['center_anchored_yx'] == [60., 53.]


def test_model_guard_rejects_wrong_task_or_roi_even_with_matching_version():
    from popcrop.annotation.analysis import ANALYSIS_VERSION, DEFAULT_DETECTOR_PARAMS
    a = analyzer()
    a.base_caller.classes = ['A', 'C', 'G', 'T']
    metadata = dict(model_type='base_caller', provenance=dict(
        analysis_version=ANALYSIS_VERSION, input_domain='raw', nuclear_diameter=55.,
        radius_scale=1.6, detector_params=DEFAULT_DETECTOR_PARAMS))
    a._validate_model_provenance(metadata, 'base')
    with pytest.raises(ValueError):
        a._validate_model_provenance(dict(metadata, model_type='blob_classifier'), 'base')
    wrong_roi = dict(metadata, provenance=dict(metadata['provenance'], nuclear_diameter=90.))
    with pytest.raises(ValueError, match='different ROI'):
        a._validate_model_provenance(wrong_roi, 'base')
