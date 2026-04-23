#!/usr/bin/env python3
"""
generate_figures.py — ThermoRG-NN paper figures
Figure 1: D-scaling law (3 panels: loss curves, residuals, R²)
Figure 2: Cooling theory β(γ) (3 panels: β vs γ, predicted vs actual, width trajectories)
Figure 3: Confounding + cross-validation (2 panels)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import os

OUT = "/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/papers/figures"
os.makedirs(OUT, exist_ok=True)
print(f"Output: {OUT}", flush=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "text.usetex": False,
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.framealpha": 0.9,
})

# ============================================================================
# FIGURE 1 — D-scaling Law (3 panels)
# ============================================================================

def plaw(D, a, b, c):
    return a * D**(-b) + c

# Data from paper: D-scaling for 3 normalization configs
configs = {
    "LN":    {"D": np.array([32, 48, 64, 96]), "loss": np.array([0.936, 0.857, 0.804, 0.736]),
               "std": np.array([0.021, 0.018, 0.015, 0.012]), "beta": 0.219, "E_floor": 0.0,
               "color": "#1f77b4"},
    "BN":    {"D": np.array([32, 48, 64, 96]), "loss": np.array([0.958, 0.800, 0.721, 0.639]),
               "std": np.array([0.021, 0.018, 0.015, 0.012]), "beta": 0.950, "E_floor": 0.466,
               "color": "#ff7f0e"},
    "None":  {"D": np.array([32, 48, 64, 96]), "loss": np.array([1.285, 1.100, 1.011, 0.926]),
               "std": np.array([0.045, 0.038, 0.031, 0.025]), "beta": 1.117, "E_floor": 0.777,
               "color": "#d62728"},
}

fits = {}
for name, cfg in configs.items():
    popt, pcov = curve_fit(plaw, cfg["D"], cfg["loss"],
                           p0=[10, 0.5, 0.5],
                           bounds=([0, 0, 0], [1000, 3, 3]), maxfev=10000)
    pred = plaw(cfg["D"], *popt)
    ss_res = ((cfg["loss"] - pred)**2).sum()
    ss_tot = ((cfg["loss"] - cfg["loss"].mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    fits[name] = {"popt": popt, "r2": r2, "E_floor": popt[2]}
    print(f"  {name}: beta={popt[1]:.3f}, E_floor={popt[2]:.3f}, R2={r2:.3f}", flush=True)

D_fit = np.linspace(25, 120, 300)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharex=True)
plt.subplots_adjust(wspace=0.35, bottom=0.22, top=0.88)

# Panel (a): D-scaling log-log
ax = axes[0]
for name, cfg in configs.items():
    popt = fits[name]["popt"]
    ax.errorbar(cfg["D"], cfg["loss"], yerr=cfg["std"],
                fmt="o", color=cfg["color"], markersize=7, capsize=3, zorder=5,
                markeredgecolor="white", markeredgewidth=0.5,
                label=f"{name}: β={popt[1]:.3f}")
    ax.plot(D_fit, plaw(D_fit, *popt), "--", color=cfg["color"], alpha=0.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Width D", fontsize=10)
ax.set_ylabel("Validation loss", fontsize=10)
ax.set_title("(a)  D-scaling: loss vs width", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.set_xticks([32, 48, 64, 96])
ax.set_xticklabels(["32", "48", "64", "96"])
ax.set_xlim(25, 120)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
ax.tick_params(axis="x", labelbottom=True)

# Panel (b): Residuals after subtracting E_floor
ax = axes[1]
for name, cfg in configs.items():
    popt = fits[name]["popt"]
    residuals = cfg["loss"] - plaw(cfg["D"], *popt)
    ax.scatter(cfg["D"], residuals, color=cfg["color"], s=60, zorder=5,
               edgecolors="white", linewidths=0.7, label=name)

ax.set_xscale("log")
ax.set_xlabel("Width D", fontsize=10)
ax.set_ylabel("Residual (data − fit)", fontsize=10)
ax.set_title("(b)  Residuals", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.tick_params(axis="x", labelbottom=False)
ax.set_xticks([32, 48, 64, 96])
ax.set_xticklabels(["32", "48", "64", "96"])
ax.set_xlim(25, 120)
ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)

# Panel (c): R² values
ax = axes[2]
names = list(configs.keys())
r2_vals = [fits[n]["r2"] for n in names]
colors = [configs[n]["color"] for n in names]
bars = ax.bar(names, r2_vals, color=colors, width=0.5, edgecolor="white", linewidth=0.7)
for bar, r2 in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{r2:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("R² goodness of fit", fontsize=10)
ax.set_title("(c)  R² goodness of fit", fontsize=10, fontweight="bold")
ax.set_ylim(0.98, 1.002)
ax.axhline(0.99, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)

fig.savefig(f"{OUT}/fig1_D_scaling_Jtopo.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 1 saved -> fig1_D_scaling_Jtopo.png", flush=True)

# ============================================================================
# FIGURE 2 — Cooling Theory β(γ) (3 panels)
# ============================================================================

gamma_c = 2.0

# Panel A: β vs γ log-linear plot
fig2a_data = {
    "LN":   {"gamma": 0.41, "beta": 0.219, "color": "#1f77b4"},
    "BN":   {"gamma": 2.29, "beta": 0.950, "color": "#ff7f0e"},
    "None": {"gamma": 3.39, "beta": 1.117, "color": "#d62728"},
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
plt.subplots_adjust(wspace=0.35, bottom=0.22, top=0.88)

ax = axes[0]
g_curve = np.linspace(0.3, 4.0, 300)
b_curve = 0.425 * np.log(g_curve / gamma_c) + 0.893

ax.fill_between(g_curve[g_curve < gamma_c], b_curve[g_curve < gamma_c],
                alpha=0.12, color="#1f77b4")
ax.fill_between(g_curve[g_curve >= gamma_c], b_curve[g_curve >= gamma_c],
                alpha=0.12, color="#d62728")
ax.plot(g_curve, b_curve, "k--", alpha=0.6, linewidth=1.5,
        label=r"$\beta(\gamma) = 0.425\ln(\gamma/\gamma_c)+0.893$")

for name, info in fig2a_data.items():
    ax.scatter(info["gamma"], info["beta"], s=130, c=info["color"], zorder=7,
               edgecolors="white", linewidths=1.2)
    ax.annotate(name, (info["gamma"], info["beta"]),
                xytext=(10, 6), textcoords="offset points",
                fontsize=9, fontweight="bold")

ax.axvline(gamma_c, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
           label=r"$\gamma_c=2.0$")
ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

ax.set_xlabel(r"Heat capacity exponent $\gamma$", fontsize=10)
ax.set_ylabel(r"Scaling exponent $\beta$", fontsize=10)
ax.set_title("(a)  β vs γ: cooling theory\n(sub- vs super-critical)", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper left")
ax.set_xlim(0, 4)
ax.set_ylim(-0.1, 1.3)

# Panel (b): Predicted vs actual β values
ax = axes[1]
gammas = [info["gamma"] for info in fig2a_data.values()]
betas  = [info["beta"]  for info in fig2a_data.values()]
colors_b = [info["color"] for info in fig2a_data.values()]
names_b  = list(fig2a_data.keys())

# Predicted from formula
b_pred = [0.425 * np.log(g / gamma_c) + 0.893 for g in gammas]

for i, (g, b, bp, c, n) in enumerate(zip(gammas, betas, b_pred, colors_b, names_b)):
    err_pct = abs(bp - b) / b * 100
    ax.scatter(b, bp, s=130, c=c, zorder=7, edgecolors="white", linewidths=1.2)
    ax.annotate(f"{n}\n({err_pct:.1f}% err)", (b, bp),
                xytext=(8, 0), textcoords="offset points",
                fontsize=8, fontweight="bold")

# Identity line
lims = [0, 1.3]
ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1.2, label="identity")
ax.set_xlabel(r"Actual $\beta$", fontsize=10)
ax.set_ylabel(r"Predicted $\beta$", fontsize=10)
ax.set_title("(b)  Predicted vs actual β\n(all within 2%)", fontsize=10, fontweight="bold")
ax.set_xlim(0, 1.3)
ax.set_ylim(0, 1.3)
ax.set_aspect("equal")
ax.legend(fontsize=8)

# Panel (c): Width-scaling trajectory (schematic: β vs D, showing cooling shift)
ax = axes[2]
D_traj = np.linspace(20, 120, 200)

for name, cfg in configs.items():
    popt = fits[name]["popt"]
    ax.plot(D_traj, plaw(D_traj, *popt), "-", color=cfg["color"],
            linewidth=1.8, label=f"{name}", alpha=0.8)
    ax.scatter(cfg["D"], cfg["loss"], color=cfg["color"], s=50, zorder=5,
              edgecolors="white", linewidths=0.7)

ax.set_xlim(20, 120)
ax.set_xlabel("Width D", fontsize=10)
ax.set_ylabel("Loss", fontsize=10)
ax.set_title("(c)  Width-scaling trajectories:\ncooling shift across configs", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.set_xticks([32, 48, 64, 96])

fig.savefig(f"{OUT}/fig2_cooling_theory.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 2 saved -> fig2_cooling_theory.png", flush=True)

# ============================================================================
# FIGURE 3 — Confounding + Cross-Validation (2 panels)
# ============================================================================

phase_b2 = np.array([
    [96, 6, 0.7739, 0.3858],
    [64, 5, 0.8062, 0.5014],
    [64, 5, 0.7838, 0.6047],
    [64, 4, 0.7538, 0.6268],
    [48, 4, 0.8027, 0.6937],
    [32, 6, 0.8627, 0.6051],
    [24, 6, 0.8774, 0.6812],
    [32, 5, 0.8455, 0.7479],
    [24, 6, 0.8727, 0.7821],
    [24, 5, 0.8701, 0.8378],
])
widths_b2 = phase_b2[:, 0]
J_b2      = phase_b2[:, 2]
loss_b2   = phase_b2[:, 3]

def get_band(w):
    if w <= 32: return "narrow"
    if w < 64: return "medium"
    return "wide"

bands = [get_band(w) for w in widths_b2]
band_colors = {"narrow": "#d62728", "medium": "#ff7f0e", "wide": "#2ca02c"}

synflow = [
    ("W=96 L=6 BN NS", 0.3527, 0.8724),
    ("W=64 L=6 BN NS", 0.3963, 0.8587),
    ("W=96 L=5 BN NS", 0.3993, 0.8607),
    ("W=64 L=5 BN NS", 0.4399, 0.8398),
    ("W=96 L=3 BN NS", 0.6670, 0.7631),
]
hbo = [
    ("W=96 L=6 BN NS", 0.3770, 0.8744),
    ("W=64 L=6 BN NS", 0.4401, 0.8486),
    ("W=96 L=5 BN NS", 0.4351, 0.8514),
    ("W=96 L=6 BN SK", 0.4582, 0.8440),
    ("W=64 L=5 BN NS", 0.5073, 0.8277),
]
random = [
    ("W=64 L=6 BN NS", 0.4270, 0.8515),
    ("W=96 L=5 BN NS", 0.4451, 0.8477),
    ("W=64 L=6 BN SK", 0.5388, 0.8149),
    ("W=96 L=6 LN NS", 0.5405, 0.8157),
    ("W=24 L=6 BN NS", 0.6643, 0.7700),
]

synflow_acc = np.array([r[2] for r in synflow])
hbo_acc     = np.array([r[2] for r in hbo])
random_acc  = np.array([r[2] for r in random])

synflow_w = np.array([96, 64, 96, 64, 96])
hbo_w     = np.array([96, 64, 96, 96, 64])
random_w  = np.array([64, 96, 64, 96, 24])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plt.subplots_adjust(wspace=0.35, bottom=0.22)

# Panel (a): J_topo vs loss, colored by width band
ax = axes[0]
for band_name in ["narrow", "medium", "wide"]:
    mask = np.array([b == band_name for b in bands])
    ax.scatter(J_b2[mask], loss_b2[mask],
               c=band_colors[band_name], s=90, label=band_name.title(),
               zorder=5, edgecolors="white", linewidths=0.7)

for band_name, color in band_colors.items():
    mask = np.array([b == band_name for b in bands])
    if sum(mask) >= 2:
        z = np.polyfit(J_b2[mask], loss_b2[mask], 1)
        p = np.poly1d(z)
        J_rng = np.linspace(J_b2[mask].min(), J_b2[mask].max(), 50)
        ax.plot(J_rng, p(J_rng), "--", color=color, alpha=0.5, linewidth=1.5)

ax.set_xlabel("J_topo", fontsize=11)
ax.set_ylabel("Validation loss", fontsize=11)
ax.set_title("(a)  J_topo vs Validation Loss", fontsize=10, fontweight="bold")
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.set_xlim(0.62, 0.92)
ax.set_ylim(0.32, 0.90)

# Panel (b): Cross-validation boxplot
ax = axes[1]
acc_data    = [random_acc, hbo_acc, synflow_acc]
width_data  = [random_w, hbo_w, synflow_w]
colors_box  = ["steelblue", "#c04040", "#40a040"]
method_names = ["Random", "ThermoRG-AL (J_topo)", "SynFlow (grad.)"]

bp = ax.boxplot(acc_data, positions=[0, 1, 2], patch_artist=True,
                widths=0.55,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(markersize=5))
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)

for i, (data, color) in enumerate(zip(acc_data, colors_box)):
    jitter = np.random.uniform(-0.12, 0.12, len(data))
    ax.scatter(np.full(len(data), i) + jitter, data, s=28, c=color,
               zorder=6, alpha=0.85, edgecolors="white", linewidths=0.5)

for i, (data, color) in enumerate(zip(acc_data, colors_box)):
    best = data.max()
    ax.text(i, best + 0.012, "best=%.3f" % best,
            ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

for i, (w_data, color) in enumerate(zip(width_data, colors_box)):
    med_w = np.median(w_data)
    ax.text(i, 0.74, "med-W=%.0f" % med_w,
            ha="center", va="top", fontsize=8, color=color)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(method_names, fontsize=9)
ax.set_ylabel("Test Accuracy (top-5 candidates)", fontsize=10)
ax.set_title("(b)  Cross-validation: SynFlow vs ThermoRG-AL", fontsize=10, fontweight="bold")
ax.set_ylim(0.70, 0.90)
ax.grid(alpha=0.3, axis="y")

fig.savefig(f"{OUT}/fig3_confounding_hbo.png", dpi=300)
plt.close()
print("Figure 3 saved -> fig3_confounding_hbo.png", flush=True)

print("\nAll figures regenerated.", flush=True)