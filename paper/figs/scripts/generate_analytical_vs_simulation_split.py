#!/usr/bin/env python3
"""Generate Analytical vs Simulation comparison as separate subfigures.
(a) Full view with all coupled configs
(b) Zoomed Pareto region with HV shading and arrows"""

import json
import sqlite3
import itertools
import csv as csv_mod
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from pathlib import Path

# --- Config ---
WORKSPACE = Path('../workspaces/benchmark_2026-01-15_231851/isscc_2023_22nm_mram_dram_half_lenet5')
DB_PATH = WORKSPACE / 'strategies.db'
ANA_PATH = WORKSPACE / 'analytical_estimates.json'
OUTPUT_DIR = Path('../paper/figs')
NUM_LAYERS = 3

# --- Shared style ---
FONTSIZE_LABEL = 10
FONTSIZE_TICK = 8
FONTSIZE_LEGEND = 8

color_sim_c = '#2874A6'
color_ana = '#C0392B'
color_reeval = '#E74C3C'


def compute_pareto(points, x_key='lat', y_key='eap'):
    if not points:
        return []
    best_by_x = {}
    for p in points:
        x = round(p[x_key], 2)
        if x not in best_by_x or p[y_key] < best_by_x[x][y_key]:
            best_by_x[x] = p
    deduped = list(best_by_x.values())
    sorted_pts = sorted(deduped, key=lambda p: p[x_key])
    pareto = [sorted_pts[0]]
    for p in sorted_pts[1:]:
        if p[y_key] < pareto[-1][y_key]:
            pareto.append(p)
    return pareto


def draw_hv_region(ax, pareto_pts, ref_point, color, alpha=0.2):
    if not pareto_pts:
        return
    x_ref, y_ref = ref_point
    points = sorted([(p['lat'], p['eap']) for p in pareto_pts])
    vertices = [(x_ref, y_ref)]
    for i in range(len(points) - 1, -1, -1):
        x, y = points[i]
        if i == len(points) - 1:
            vertices.append((x_ref, y))
        else:
            vertices.append((points[i + 1][0], y))
        vertices.append((x, y))
    vertices.append((points[0][0], y_ref))
    poly = Polygon(vertices, facecolor=color, alpha=alpha, edgecolor='none', zorder=1)
    ax.add_patch(poly)


def main():
    # Load data (same as original)
    with open(ANA_PATH) as f:
        ana_estimates = json.load(f)

    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row

    sim_per_layer, ana_per_layer = {}, {}
    for layer in range(NUM_LAYERS):
        rows = db.execute('''SELECT strategy_id, latency_ns, energy_nj,
            ibuf_area_mm2, obuf_area_mm2, input_tile_p, input_tile_q,
            output_tile_p, output_tile_q
            FROM strategy_results WHERE layer_index=?''', (layer,)).fetchall()
        sim_coupled, ana_coupled = {}, {}
        for r in rows:
            if r['input_tile_p'] != r['output_tile_p'] or r['input_tile_q'] != r['output_tile_q']:
                continue
            sid = r['strategy_id']
            sim_coupled[sid] = {
                'lat': r['latency_ns'], 'energy': r['energy_nj'],
                'ibuf': r['ibuf_area_mm2'], 'obuf': r['obuf_area_mm2'],
            }
            key = f"{layer}_{sid}"
            if key in ana_estimates:
                ana_coupled[sid] = {
                    'lat': ana_estimates[key]['latency_ns'],
                    'energy': ana_estimates[key]['energy_nj'],
                    'ibuf': r['ibuf_area_mm2'], 'obuf': r['obuf_area_mm2'],
                }
        sim_per_layer[layer] = sim_coupled
        ana_per_layer[layer] = ana_coupled
    db.close()

    common_sids = {}
    for layer in range(NUM_LAYERS):
        common_sids[layer] = sorted(set(sim_per_layer[layer].keys()) & set(ana_per_layer[layer].keys()))

    sim_points, ana_points = [], []
    for combo in itertools.product(*[common_sids[l] for l in range(NUM_LAYERS)]):
        sim_lat = sum(sim_per_layer[l][sid]['lat'] for l, sid in enumerate(combo))
        sim_energy = sum(sim_per_layer[l][sid]['energy'] for l, sid in enumerate(combo))
        sim_ibuf = max(sim_per_layer[l][sid]['ibuf'] for l, sid in enumerate(combo))
        sim_obuf = max(sim_per_layer[l][sid]['obuf'] for l, sid in enumerate(combo))
        sim_eap = sim_energy * (sim_ibuf + sim_obuf)

        ana_lat = sum(ana_per_layer[l][sid]['lat'] for l, sid in enumerate(combo))
        ana_energy = sum(ana_per_layer[l][sid]['energy'] for l, sid in enumerate(combo))
        ana_ibuf = max(ana_per_layer[l][sid]['ibuf'] for l, sid in enumerate(combo))
        ana_obuf = max(ana_per_layer[l][sid]['obuf'] for l, sid in enumerate(combo))
        ana_eap = ana_energy * (ana_ibuf + ana_obuf)

        sim_points.append({'lat': sim_lat, 'eap': sim_eap, 'key': combo})
        ana_points.append({'lat': ana_lat, 'eap': ana_eap, 'key': combo})

    # Load CSV Pareto
    comp_dir = WORKSPACE / 'plots' / 'network_full' / 'comparison'
    obj_pair = 'latency_ns vs buffer_eap'

    def load_csv_pareto(path, pair):
        pts = []
        with open(path) as f:
            for row in csv_mod.DictReader(f):
                if row['objective_pair'].strip('"') == pair:
                    pts.append({
                        'lat': float(row['latency_ns']),
                        'eap': float(row['buffer_eap']),
                        'key': (int(row['0_strategy']), int(row['1_strategy']), int(row['2_strategy'])),
                    })
        return pts

    sim_pareto = load_csv_pareto(comp_dir / 'coupled.csv', obj_pair)
    ana_pareto = load_csv_pareto(comp_dir / 'analytical_estimated.csv', obj_pair)
    ana_reeval_pareto = load_csv_pareto(comp_dir / 'analytical.csv', obj_pair)

    sim_by_key = {p['key']: p for p in sim_points}
    arrows = []
    for ap in ana_pareto:
        sp = sim_by_key.get(ap['key'])
        if sp:
            arrows.append({
                'ana_lat': ap['lat'], 'ana_eap': ap['eap'],
                'sim_lat': sp['lat'], 'sim_eap': sp['eap'],
            })

    # Load HV
    three_way_path = WORKSPACE.parent / 'three_way_hv_comparison.json'
    with open(three_way_path) as f:
        three_way = json.load(f)
    ws_key = WORKSPACE.name
    hv_data = three_way['workspaces'][ws_key]['Buffer EAP']
    hv_sim_norm = hv_data['hv_coupled']
    hv_ana_norm = hv_data['hv_analytical']
    improvement = hv_data['coupled_vs_analytical_pct']

    zoom_xlim = (595893, 1475211)
    zoom_ylim = (14.92, 56.59)

    # --- Helper ---
    def plot_common(ax, show_hv=False, show_pareto=False, show_reeval=False, xlim=None, ylim=None, show_legend=True):
        ax.scatter([p['lat'] for p in sim_points], [p['eap'] for p in sim_points],
                   s=25, c='#A8C8E8', edgecolors='#5B9BD5', linewidth=0.4, alpha=0.7,
                   zorder=2, label='Proposed')
        ax.scatter([p['lat'] for p in ana_points], [p['eap'] for p in ana_points],
                   s=25, c='#F5B7B1', edgecolors='#E07060', linewidth=0.4, alpha=0.7,
                   zorder=2, label='Conventional')

        if show_pareto:
            if show_hv:
                all_hv_lats = [p['lat'] for p in sim_pareto + ana_reeval_pareto]
                all_hv_eaps = [p['eap'] for p in sim_pareto + ana_reeval_pareto]
                hv_ref_pt = (max(all_hv_lats) * 1.05, max(all_hv_eaps) * 1.05)
                draw_hv_region(ax, sim_pareto, hv_ref_pt, color_sim_c, alpha=0.15)
                draw_hv_region(ax, ana_reeval_pareto, hv_ref_pt, color_ana, alpha=0.15)

            ap = sorted(ana_pareto, key=lambda p: p['lat'])
            ax.scatter([p['lat'] for p in ap], [p['eap'] for p in ap],
                       alpha=0.8, s=40, c=color_ana, marker='*', zorder=4,
                       label='Conventional Pareto')
            ax.plot([p['lat'] for p in ap], [p['eap'] for p in ap],
                    '--', color=color_ana, alpha=0.5, linewidth=1.2, zorder=3)

            sp = sorted(sim_pareto, key=lambda p: p['lat'])
            ax.scatter([p['lat'] for p in sp], [p['eap'] for p in sp],
                       alpha=0.8, s=40, c=color_sim_c, marker='*', zorder=5,
                       label='Proposed Pareto')
            ax.plot([p['lat'] for p in sp], [p['eap'] for p in sp],
                    '--', color=color_sim_c, alpha=0.5, linewidth=1.2, zorder=4)

        if show_reeval:
            for a in arrows:
                ax.annotate('', xy=(a['sim_lat'], a['sim_eap']),
                            xytext=(a['ana_lat'], a['ana_eap']),
                            arrowprops=dict(arrowstyle='->', color=color_reeval,
                                            linewidth=1.8, shrinkA=4, shrinkB=4),
                            zorder=6)
            if arrows:
                ax.scatter([a['sim_lat'] for a in arrows], [a['sim_eap'] for a in arrows],
                           s=60, marker='^', c=color_reeval, edgecolors='black',
                           linewidth=0.8, zorder=7, label='Conventional, re-evaluated')

        ax.set_xlabel('Latency (ns)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('EAP (mm²·nJ)', fontsize=FONTSIZE_LABEL)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.4)
        if show_legend:
            ax.legend(fontsize=FONTSIZE_LEGEND, loc='upper right', framealpha=0.9, fancybox=False)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    # --- (a) Full view ---
    fig_a, ax_a = plt.subplots(1, 1, figsize=(3.5, 3.0))
    plot_common(ax_a, show_hv=False, show_pareto=True, show_reeval=False, show_legend=False)
    rect = Rectangle((zoom_xlim[0], zoom_ylim[0]),
                      zoom_xlim[1] - zoom_xlim[0], zoom_ylim[1] - zoom_ylim[0],
                      linewidth=1.5, edgecolor='#333333', facecolor='#f0f0f0',
                      alpha=0.3, linestyle='--', zorder=8)
    ax_a.add_patch(rect)
    # Manual legend with all 5 entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F5B7B1', markeredgecolor='#E07060', markersize=8, label='Conventional'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=color_ana, markersize=12, label='Conventional Pareto'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=color_reeval, markeredgecolor='black', markersize=8, label='Conventional, re-evaluated'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#A8C8E8', markeredgecolor='#5B9BD5', markersize=8, label='Proposed'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=color_sim_c, markersize=12, label='Proposed Pareto'),
    ]
    ax_a.legend(handles=legend_elements, fontsize=FONTSIZE_LEGEND, loc='upper right', framealpha=0.9, fancybox=False)
    fig_a.tight_layout()
    fig_a.savefig(OUTPUT_DIR / 'fig12_a.pdf', dpi=300, bbox_inches='tight')
    fig_a.savefig(OUTPUT_DIR / 'fig12_a.png', dpi=200, bbox_inches='tight')
    plt.close(fig_a)
    print(f"Saved: analytical_vs_simulation_a.pdf")

    # --- (b) Zoomed Pareto region ---
    fig_b, ax_b = plt.subplots(1, 1, figsize=(3.5, 3.0))
    plot_common(ax_b, show_hv=True, show_pareto=True, show_reeval=True, xlim=zoom_xlim, ylim=zoom_ylim, show_legend=False)

    # Two-column HV legend: left labels flush-left, right values flush-right.
    rows = [
        ('Conventional, re-evaluated', f'HV = {hv_ana_norm:.3f}'),
        ('Proposed',                   f'HV = {hv_sim_norm:.3f}'),
        ('Improvement',                f'+{improvement:.1f}%'),
    ]
    box_x0, box_x1 = 0.03, 0.88
    box_y1 = 0.97
    line_h = 0.055
    pad_x = 0.012
    pad_y = 0.012
    text_kw = dict(fontsize=7, color='black',
                   transform=ax_b.transAxes, zorder=11)
    for i, (left, right) in enumerate(rows):
        y = box_y1 - pad_y - line_h * (i + 0.5)
        ax_b.text(box_x0 + pad_x,   y, left,  ha='left',  va='center', **text_kw)
        ax_b.text(box_x1 - pad_x,   y, right, ha='right', va='center', **text_kw)
    from matplotlib.patches import FancyBboxPatch
    box_h = line_h * len(rows) + 2 * pad_y
    frame = FancyBboxPatch((box_x0, box_y1 - box_h), box_x1 - box_x0, box_h,
                           transform=ax_b.transAxes,
                           boxstyle='square,pad=0.0',
                           facecolor='white', edgecolor='0.8',
                           alpha=0.9, linewidth=0.8, zorder=10)
    ax_b.add_patch(frame)
    fig_b.tight_layout()
    fig_b.savefig(OUTPUT_DIR / 'fig12_b.pdf', dpi=300, bbox_inches='tight')
    fig_b.savefig(OUTPUT_DIR / 'fig12_b.png', dpi=200, bbox_inches='tight')
    plt.close(fig_b)
    print(f"Saved: analytical_vs_simulation_b.pdf")


if __name__ == '__main__':
    main()
