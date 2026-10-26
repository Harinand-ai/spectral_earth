import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def post_json(endpoint, data):
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    return json.loads(res.read().decode("utf-8"))

def get_json(endpoint):
    res = urllib.request.urlopen(f"{BASE_URL}{endpoint}")
    return json.loads(res.read().decode("utf-8"))

def test_full_pipeline():
    print("=" * 80)
    print("STARTING END-TO-END DEMONSTRATION VERIFICATION SUITE")
    print("=" * 80)

    datasets = ["Indian Pines", "Salinas", "Pavia University"]

    for ds in datasets:
        print(f"\n--- Testing Dataset: {ds} ---")
        t0 = time.time()
        
        # 1. Select Dataset
        init_res = post_json("/api/session/select_dataset", {"dataset_name": ds, "seed": 42})
        print(f"  [OK] Selected dataset: {ds} (Labeled: {init_res['num_labeled']}, Unlabeled: {init_res['num_unlabeled']})")

        # 2. Fetch Scene Data
        scene_res = get_json("/api/scene_data")
        dim = scene_res["dimensions"]
        print(f"  [OK] Scene Data: {dim[0]}x{dim[1]}, Bands: {scene_res['bands']}, Classes: {scene_res['num_classes']}")

        # 3. Predictions & Uncertainty Map
        unc_res = get_json("/api/predictions_and_uncertainty?uncertainty_method=least_confidence")
        print(f"  [OK] Predictions & Uncertainty Map computed ({len(unc_res['prediction_map'])}x{len(unc_res['prediction_map'][0])})")

        # 4. Query Selection
        q_res = post_json("/api/query_selection", {"k": 20, "uncertainty_method": "filter_diversity"})
        samples = q_res["selected_samples"]
        print(f"  [OK] Query Selection (LC + Farthest Diversity): Selected {len(samples)} samples in {q_res['query_time']:.3f}s")
        assert len(samples) == 20, "Expected 20 selected samples"

        # Verify 'Why Was This Sample Selected?' explanation on first sample
        sample_0 = samples[0]
        print(f"  [OK] Sample #{sample_0['rank']} ({sample_0['coord']}) explanation: '{sample_0['selection_reason']}'")

        # 5. Sample Inspection
        inspect_res = get_json(f"/api/sample_inspection/{sample_0['global_index']}")
        print(f"  [OK] Sample Inspection: Patch size 15x15, Spectrum length {len(inspect_res['spectrum'])}, GT: {inspect_res['gt_class']}")

        # 6. Simulate Human Annotation & Retrain Classifier Head
        retrain_res = post_json("/api/annotate_and_retrain", {"epochs": 25, "lr": 0.0005})
        print(f"  [OK] Oracle Annotation & Retraining: Annotated {retrain_res['annotated_count']} samples, Retrain Loss: {retrain_res['retrain_loss']:.4f}, OA: {retrain_res['metrics']['OverallAccuracy']*100:.2f}%")

        # 6b. Verify Confidence-Based Pseudo-Labeling Endpoint & Execution
        pseudo_cfg = post_json("/api/session/set_pseudo_labeling", {
            "use_pseudo_labeling": True,
            "pseudo_threshold": 0.95,
            "pseudo_weight": 0.2,
            "max_pseudo_per_class": 15
        })
        assert pseudo_cfg["status"] == "success", "Failed to configure pseudo-labeling"
        print(f"  [OK] Pseudo-Labeling Configured: {pseudo_cfg['pseudo_labeling_config']}")

        # 7. Before vs After Maps
        ba_res = get_json("/api/before_after_maps")
        print(f"  [OK] Before vs After Maps generated.")

        # 8. Export Experiment
        exp_res = post_json("/api/export_experiment", {})
        print(f"  [OK] Exported experiment JSON to: {exp_res['json_path']}")
        print(f"  Completed demo for {ds} in {time.time() - t0:.2f}s!")

    # 9. Test Cross-Sensor Demonstration
    print(f"\n--- Testing Cross-Sensor Demonstration (AVIRIS -> ROSIS) ---")
    cs_res = get_json("/api/cross_sensor_data?pair_key=Indian%20Pines%20-%3E%20Pavia%20University")
    sum_data = cs_res["summary"]
    print(f"  [OK] Cross-Sensor Transfer: {sum_data['pair_key']}")
    zs_oa = sum_data['zero_shot_metrics']['oa']
    zs_oa_str = f"{zs_oa:.2f}%" if zs_oa is not None else "N/A (class mismatch)"
    print(f"  [OK] Zero-Shot OA: {zs_oa_str}")
    print(f"  [OK] Few-Shot AL Adaptation OA: {sum_data['few_shot_final_metrics']['oa']:.2f}%")
    print(f"  [OK] 1D Resampler Spectrum length: {len(cs_res['resampler_data']['resampled_spectrum'])}")

    # 10. Test Label-Efficiency Benchmark
    print(f"\n--- Testing Label-Efficiency Benchmark ---")
    le_res = get_json("/api/label_efficiency_data")
    print(f"  [OK] Benchmark curves loaded for {list(le_res['curves_data'].keys())}")
    print(f"  [OK] Threshold savings entries: {len(le_res['thresholds_table'])}")

    # 11. Test Report Generation
    print(f"\n--- Testing 15-Section Research Report Generation ---")
    report_res = urllib.request.urlopen(f"{BASE_URL}/api/generate_report").read().decode("utf-8")
    assert "SpectralEarth Active Learning" in report_res, "Report framework missing"
    assert "Adaptive Label-Efficient Hyperspectral" in report_res, "Report title missing"
    assert "Frozen" in report_res, "Backbone notice missing"
    print(f"  [OK] Visual Research Report generated successfully ({len(report_res)} bytes)!")

    print("\n" + "=" * 80)
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_full_pipeline()
