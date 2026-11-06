"""
ConformAL-HSI: Run Confirmation Suite and LayerNorm Diagnostic Suite,
then generate all final research validation artifacts and figures.
"""

import os
import sys
import subprocess

PYTHON_EXE = sys.executable

def run_cmd(cmd_args, desc):
    print("=" * 80)
    print(f">>> {desc}")
    print("Command:", " ".join(cmd_args))
    print("=" * 80)
    res = subprocess.run(cmd_args, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Step failed with code {res.returncode}: {desc}")

def main():
    # 1. Multi-Seed Confirmation (seeds 45, 46 for Head B and Head C)
    run_cmd([
        PYTHON_EXE, "-u", "run_classifier_capacity_interaction_study.py",
        "--heads", "head_b", "head_c",
        "--seeds", "45", "46",
        "--norm_type", "batchnorm",
        "--out_dir", "results/final_research_validation",
        "--tag", "confirmation"
    ], "STEP 1: Multi-Seed Confirmation (Seeds 45, 46 for Head B & Head C)")

    # 2. BatchNorm vs LayerNorm Diagnostic (seeds 42, 43, 44 for Head B, Head C, Head D with layernorm)
    run_cmd([
        PYTHON_EXE, "-u", "run_classifier_capacity_interaction_study.py",
        "--heads", "head_b", "head_c", "head_d",
        "--seeds", "42", "43", "44",
        "--norm_type", "layernorm",
        "--out_dir", "results/final_research_validation",
        "--tag", "layernorm"
    ], "STEP 2: LayerNorm Diagnostic Suite (Heads B, C, D with LayerNorm)")

    # 3. Generate All Master CSVs and 10 Publication Figures
    run_cmd([
        PYTHON_EXE, "-u", "generate_final_research_validation_artifacts.py"
    ], "STEP 3: Generate Master Tables, Statistical Models & 10 Publication Figures")

    print("\n[COMPLETE] All confirmation, diagnostic, synthesis, and figure tasks finished!")

if __name__ == "__main__":
    main()
