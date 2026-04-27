#!/usr/bin/env python3
"""
Generate Figure 5: Analytical Model Prediction Error Heatmap for paper.

Simplified version without legacy strategy markers (LS1-LS5).
Compact stats display with larger fonts for print readability.
"""

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Configuration
BENCHMARK_PATH = Path(__file__).parent.parent.parent.parent / "workspaces" / "benchmark_2026-01-15_111301"
OUTPUT_PATH = Path(__file__).parent.parent / "prediction_error_heatmap.png"
OUTPUT_PATH_PDF = Path(__file__).parent.parent / "prediction_error_heatmap.pdf"

# Target: MRAM hardware, VGG-11 network, Layer 0
HARDWARE = "isscc_2023_22nm_mram_dram_half"
NETWORK = "vgg11"
LAYER_INDEX = 0


def find_workspace():
    """Find the workspace for the specified hardware and network."""
    benchmark_path = BENCHMARK_PATH

    # Look for matching workspace
    for ws_dir in benchmark_path.iterdir():
        if not ws_dir.is_dir():
            continue
        if HARDWARE in ws_dir.name and NETWORK in ws_dir.name:
            return ws_dir

    raise FileNotFoundError(f"No workspace found for {HARDWARE}/{NETWORK}")


def load_analytical_error_data(workspace_path: Path, layer_index: int = 0):
    """Load simulation results and compute analytical errors."""
    # Add src/python to path for imports
    import sys
    src_path = Path(__file__).parent.parent.parent.parent / "src" / "python"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from analysis.analytical_model import (
        AnalyticalModel,
        HardwareConfig,
        LayerConfig,
        TilingConfig,
    )

    workspace_path = Path(workspace_path)

    # Load configs
    hw_config = HardwareConfig.from_config(
        json.loads((workspace_path / "hardware_config.json").read_text())
    )
    layer_config_path = workspace_path / "layers" / f"L{layer_index}.json"
    layer_data = json.loads(layer_config_path.read_text())
    layer = LayerConfig.from_dict(layer_data["params"])

    model = AnalyticalModel(hw_config)

    # Load simulation results
    db_path = workspace_path / "strategies.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT strategy_id, tiling_config, latency_ns, energy_nj
        FROM strategy_results
        WHERE layer_index = ?
        """,
        (layer_index,),
    )

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        tiling_raw = row["tiling_config"]
        if isinstance(tiling_raw, str):
            tiling_config = json.loads(tiling_raw)
            if isinstance(tiling_config, str):
                tiling_config = json.loads(tiling_config)
        else:
            tiling_config = tiling_raw

        sim_latency = row["latency_ns"]
        sim_energy = row["energy_nj"]

        if sim_latency is None or sim_latency <= 0:
            continue
        if sim_energy is None or sim_energy <= 0:
            continue

        try:
            tiling = TilingConfig.from_dict(tiling_config, layer)
            est = model.estimate(layer, tiling)
            est_latency = est.latency_ns

            latency_error = abs(est_latency - sim_latency) / sim_latency * 100

            results.append({
                "strategy_id": row["strategy_id"],
                "output_tile_size": tiling_config.get("output_tile_p", 0) * tiling_config.get("output_tile_q", 0),
                "input_tile_size": tiling_config.get("input_tile_p", 0) * tiling_config.get("input_tile_q", 0),
                "latency_error_pct": latency_error,
            })
        except Exception:
            continue

    return results


def main():
    # Find workspace
    workspace = find_workspace()
    print(f"Using workspace: {workspace}")

    # Load data
    data = load_analytical_error_data(workspace, LAYER_INDEX)
    print(f"Loaded {len(data)} data points")

    if not data:
        raise ValueError("No data found")

    # Extract arrays
    x = np.array([d["output_tile_size"] for d in data])
    y = np.array([d["input_tile_size"] for d in data])
    values = np.array([d["latency_error_pct"] for d in data])

    # Print stats
    print(f"\nLatency Error Statistics:")
    print(f"  Min:    {values.min():.1f}%")
    print(f"  Max:    {values.max():.1f}%")
    print(f"  Mean:   {values.mean():.1f}%")
    print(f"  Median: {np.median(values):.1f}%")

    # Create figure with larger size for readability
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Color normalization (percentile-based for better contrast)
    p2 = np.percentile(values, 2)
    p98 = np.percentile(values, 98)
    vmin = max(0, p2 - 0.1 * (p98 - p2))
    vmax = p98 + 0.1 * (p98 - p2)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Scatter plot - larger markers
    scatter = ax.scatter(
        x, y, c=values,
        cmap="RdYlGn_r",  # Red=high error, Green=low error
        norm=norm,
        s=50,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
    )

    # Colorbar with larger font
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Latency Error (%)", fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Axis labels
    ax.set_xlabel(r"Output Tile Size ($t_{out,p} \times t_{out,q}$)", fontsize=13, fontweight='bold')
    ax.set_ylabel(r"Input Tile Size ($t_{in,p} \times t_{in,q}$)", fontsize=13, fontweight='bold')
    # Title omitted - use LaTeX caption instead

    # Log scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Tick label size
    ax.tick_params(axis='both', labelsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_PATH_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Saved: {OUTPUT_PATH_PDF}")


if __name__ == "__main__":
    main()
