import os
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


def set_publication_style():
    """Configure matplotlib with publication-ready aesthetic defaults."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 7,
    })


def plot_learning_curves(
    history: List[Dict[str, Any]],
    save_path: str = "logs/visualizations/learning_curves.png",
    title_suffix: str = "Indian Pines",
):
    """
    Plot OA, AA, F1 Score, and Kappa Coefficient across Active Learning rounds.
    """
    set_publication_style()
    rounds = [entry["round"] for entry in history]
    n_labeled = [entry["num_labeled"] for entry in history]
    
    oa = [entry.get("metrics", {}).get("OverallAccuracy", 0.0) * 100 for entry in history]
    aa = [entry.get("metrics", {}).get("AverageAccuracy", 0.0) * 100 for entry in history]
    f1 = [entry.get("metrics", {}).get("F1Score", 0.0) * 100 for entry in history]
    kappa = [entry.get("metrics", {}).get("Kappa", 0.0) * 100 for entry in history]

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.plot(rounds, oa, 'o-', color='#1f77b4', label='Overall Accuracy (OA %)')
    ax1.plot(rounds, aa, 's--', color='#2ca02c', label='Average Accuracy (AA %)')
    ax1.plot(rounds, f1, '^-.', color='#ff7f0e', label='F1 Score (%)')
    ax1.plot(rounds, kappa, 'd:', color='#9467bd', label='Kappa (%)')

    ax1.set_xlabel('Active Learning Round')
    ax1.set_ylabel('Performance (%)')
    ax1.set_title(f'Active Learning Progression ({title_suffix})')
    ax1.set_xticks(rounds)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', frameon=True)

    # Secondary x-axis showing number of labeled samples
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(rounds)
    ax2.set_xticklabels(n_labeled)
    ax2.set_xlabel('Labeled Pool Size (N)')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved publication figure: {save_path}")


def plot_strategy_comparison(
    strategy_results: Dict[str, List[Dict[str, Any]]],
    metric_key: str = "OverallAccuracy",
    save_path: str = "logs/visualizations/strategy_comparison.png",
    dataset_name: str = "Indian Pines",
):
    """
    Plot comparative performance curves across different Query Strategies.
    """
    set_publication_style()
    colors = {
        "least_confidence": "#1f77b4",
        "entropy": "#ff7f0e",
        "margin": "#2ca02c",
        "adaptive": "#d62728",
        "hybrid": "#9467bd",
        "diversity": "#8c564b",
        "random": "#7f7f7f",
    }
    markers = {
        "least_confidence": "o",
        "entropy": "s",
        "margin": "^",
        "adaptive": "D",
        "hybrid": "v",
        "diversity": "p",
        "random": "X",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for strat_name, history in strategy_results.items():
        rounds = [entry["round"] for entry in history]
        vals = [entry.get("metrics", {}).get(metric_key, 0.0) * 100 for entry in history]
        
        c = colors.get(strat_name.lower(), None)
        m = markers.get(strat_name.lower(), 'o')
        
        ax.plot(rounds, vals, label=strat_name.capitalize(), marker=m, color=c)

    ax.set_xlabel('Active Learning Round')
    ax.set_ylabel(f'{metric_key} (%)')
    ax.set_title(f'Query Strategy Comparison — {dataset_name} ({metric_key})')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', frameon=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved publication figure: {save_path}")


def plot_loss_and_calibration(
    history: List[Dict[str, Any]],
    save_path: str = "logs/visualizations/loss_and_calibration.png",
):
    """
    Plot training loss and Expected Calibration Error (ECE) across AL rounds.
    """
    set_publication_style()
    rounds = [entry["round"] for entry in history]
    losses = [entry["retrain_loss"] for entry in history]
    ece_vals = [entry.get("metrics", {}).get("ECE", 0.0) * 100 for entry in history]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color = '#d62728'
    ax1.set_xlabel('Active Learning Round')
    ax1.set_ylabel('Retraining CrossEntropy Loss', color=color)
    ax1.plot(rounds, losses, color=color, marker='o', label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#1f77b4'
    ax2.set_ylabel('Expected Calibration Error (ECE %)', color=color)
    ax2.plot(rounds, ece_vals, color=color, marker='s', linestyle='--', label='ECE (%)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Loss & Probability Calibration across AL Rounds')
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved publication figure: {save_path}")
