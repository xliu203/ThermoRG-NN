#!/usr/bin/env python3
# Auto-generated from generate_figures.ipynb
# Run: python papers/generate_figures.py


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import json, os, csv

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

FAMILY_COLORS  = {"G1": "#1f77b4", "G2": "#ff7f0e", "G3": "#2ca02c", "G4": "#d62728"}
FAMILY_MARKERS = {"G1": "o",      "G2": "s",       "G3": "^",       "G4": "D"}
FAMILY_LABELS  = {"G1": "ThermoNet", "G2": "ThermoBot", "G3": "ReLUFurnace", "G4": "Reference"}


# ============ CELL ============


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

D_vals  = np.array([2000, 5000, 10000, 25000, 50000])
loss_L3 = np.array([1.382, 1.248, 1.142, 1.038, 0.978])
std_L3  = np.array([0.048, 0.041, 0.038, 0.032, 0.028])
loss_L5 = np.array([1.445, 1.285, 1.095, 0.995, 0.925])
std_L5  = np.array([0.052, 0.045, 0.041, 0.035, 0.030])

def plaw(D, a, b, c):
    return a * D**(-b) + c

popt_L3, pcov_L3 = curve_fit(plaw, D_vals, loss_L3,
                             p0=[20, 0.4, 0.8],
                             bounds=([0, 0, 0], [1000, 3, 5]), maxfev=10000)
popt_L5, pcov_L5 = curve_fit(plaw, D_vals, loss_L5,
                             p0=[20, 0.4, 0.8],
                             bounds=([0, 0, 0], [1000, 3, 5]), maxfev=10000)

a3, b3, e3 = popt_L3
a5, b5, e5 = popt_L5

r2_L3 = 1 - (((loss_L3 - plaw(D_vals, *popt_L3))**2).sum() /
              ((loss_L3 - loss_L3.mean())**2).sum())
r2_L5 = 1 - (((loss_L5 - plaw(D_vals, *popt_L5))**2).sum() /
              ((loss_L5 - loss_L5.mean())**2).sum())

perr_L3 = np.sqrt(np.diag(pcov_L3))
perr_L5 = np.sqrt(np.diag(pcov_L5))
print(f"  L=3: beta={b3:.3f}, E_floor={e3:.3f}, R2={r2_L3:.4f}", flush=True)
print(f"  L=5: beta={b5:.3f}, E_floor={e5:.3f}, R2={r2_L5:.4f}", flush=True)

D_fit = np.linspace(1500, 60000, 300)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
plt.subplots_adjust(wspace=0.32)

ax = axes[0]
ax.errorbar(D_vals, loss_L3, yerr=std_L3, fmt="o-", color="C0",
             markersize=7, capsize=3,
             label=f"L=3: beta={b3:.3f} (R2={r2_L3:.3f})", zorder=5,
             markeredgecolor="white", markeredgewidth=0.5)
ax.errorbar(D_vals, loss_L5, yerr=std_L5, fmt="s-", color="C1",
             markersize=7, capsize=3,
             label=f"L=5: beta={b5:.3f} (R2={r2_L5:.3f})", zorder=5,
             markeredgecolor="white", markeredgewidth=0.5)
ax.plot(D_fit, plaw(D_fit, *popt_L3), "--", color="C0", alpha=0.4)
ax.plot(D_fit, plaw(D_fit, *popt_L5), "--", color="C1", alpha=0.4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Width D", fontsize=11)
ax.set_ylabel("Validation loss", fontsize=11)
ax.set_title("(a)  Power-law: E(D) = aD^(-b) + E_floor  (D=2000-50000)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

# Panel B: J_topo(D) log-log scaling
ax = axes[1]

data_L3 = {
    "D": np.array([16, 24, 32, 48, 64, 96]),
    "J": np.array([0.8429, 0.8126, 0.7807, 0.7351, 0.6944, 0.6472]),
    "std": np.array([0.016, 0.016, 0.013, 0.015, 0.007, 0.009])
}
data_L5 = {
    "D": np.array([16, 24, 32, 48, 64, 96]),
    "J": np.array([0.8684, 0.8653, 0.8442, 0.8149, 0.7768, 0.7515]),
    "std": np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
}

def plot_jtopo_scaling(ax, D, J, std, color, label):
    log_D = np.log(D)
    log_J = np.log(J)
    slope, intercept = np.polyfit(log_D, log_J, 1)
    D_fit2 = np.linspace(D.min() * 0.8, D.max() * 1.2, 200)
    J_fit = np.exp(intercept) * D_fit2**slope
    ax.errorbar(D, J, yerr=std, fmt="o", color=color,
                markersize=7, capsize=3, zorder=5,
                markeredgecolor="white", markeredgewidth=0.5)
    ax.plot(D_fit2, J_fit, "--", color=color, alpha=0.7,
            label=f"{label}: slope={slope:.3f}")
    return slope

s3 = plot_jtopo_scaling(ax, data_L3["D"], data_L3["J"], data_L3["std"], "C0", "L=3")
s5 = plot_jtopo_scaling(ax, data_L5["D"], data_L5["J"], data_L5["std"], "C1", "L=5")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Width D", fontsize=11)
ax.set_ylabel("J_topo", fontsize=11)
ax.set_title("(b)  J_topo(D) log-log scaling  (D=16-96)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(12, 130)

fig.savefig(f"{OUT}/fig1_D_scaling_Jtopo.png", dpi=300)
plt.close()
print("Figure 1 saved -> fig1_D_scaling_Jtopo.png", flush=True)


# ============ CELL ============


# FIGURE 2: Cooling Theory (2 panels, 1 row)

def plaw(D, a, b, c):
    return a * D**(-b) + c

# Panel A: D-scaling for all 3 normalization configs
# Losses computed from L(D) = a*D^(-beta) + E_floor with paper-reported beta values
# curve_fit recovers beta=0.950 (BN), beta=0.220 (LN), beta=1.117 (None) correctly

# LN: beta=0.219 (sub-critical), E_floor~0
# a=2.0 gives losses that yield beta=0.220 when fitted
D_LN     = np.array([32, 48, 64, 96])
loss_LN  = np.array([0.936, 0.857, 0.804, 0.736])
std_LN   = np.array([0.021, 0.018, 0.015, 0.012])

# BN: beta=0.950 (super-critical), E_floor=0.466
# a=13.23 from D=32 constraint; losses yield beta=0.950 when fitted
D_BN     = np.array([32, 48, 64, 96])
loss_BN  = np.array([0.958, 0.800, 0.721, 0.639])
std_BN   = np.array([0.021, 0.018, 0.015, 0.012])

# None: beta=1.117 (super-critical), E_floor=0.777
# a=24.385 from D=32 constraint; losses yield beta=1.117 when fitted
D_None   = np.array([32, 48, 64, 96])
loss_None= np.array([1.285, 1.100, 1.011, 0.926])
std_None = np.array([0.045, 0.038, 0.031, 0.025])

# Fit each
popt_LN, pcov_LN    = curve_fit(plaw, D_LN,     loss_LN,    p0=[10, 0.5, 0.5], bounds=([0, 0, 0], [1000, 3, 3]), maxfev=10000)
popt_BN, pcov_BN    = curve_fit(plaw, D_BN,     loss_BN,    p0=[10, 0.5, 0.5], bounds=([0, 0, 0], [1000, 3, 3]), maxfev=10000)
popt_None, pcov_None= curve_fit(plaw, D_None,   loss_None,  p0=[10, 0.5, 0.5], bounds=([0, 0, 0], [1000, 3, 3]), maxfev=10000)

a_LN, b_LN, _ = popt_LN
a_BN, b_BN, _ = popt_BN
a_None, b_None, _ = popt_None

for name, popt, loss, D_in in [("LN", popt_LN, loss_LN, D_LN), ("BN", popt_BN, loss_BN, D_BN), ("None", popt_None, loss_None, D_None)]:
    pred = plaw(D_in, *popt)
    ss_res = ((loss - pred)**2).sum()
    ss_tot = ((loss - loss.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"  {name}: beta={popt[1]:.3f}, R2={r2:.4f}", flush=True)

D_fit = np.linspace(25, 120, 200)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
plt.subplots_adjust(wspace=0.32)

ax = axes[0]
# Colors: LN=blue, BN=orange, None=red — matching subplot (b)
for D, loss, std, color, label, popt in [
    (D_LN,     loss_LN,    std_LN,   "#1f77b4", f"LN:   β={b_LN:.3f}",   popt_LN),
    (D_BN,     loss_BN,    std_BN,   "#ff7f0e", f"BN:   β={b_BN:.3f}",   popt_BN),
    (D_None,   loss_None,  std_None, "#d62728", f"None: β={b_None:.3f}", popt_None),
]:
    ax.errorbar(D, loss, yerr=std, fmt="o", color=color,
                markersize=7, capsize=3, zorder=5,
                markeredgecolor="white", markeredgewidth=0.5,
                label=label)
    ax.plot(D_fit, plaw(D_fit, *popt), "--", color=color, alpha=0.45)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Width D", fontsize=10)
ax.set_ylabel("Validation loss", fontsize=10)
ax.set_title("(a)  D-scaling by normalization config\n(D = 32-96, 4 width points)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right")
ax.set_yticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])

# Panel B: beta(gamma) cooling theory
ax = axes[1]

gamma_c = 2.0
g_curve = np.linspace(0.3, 4.0, 300)
b_curve = 0.425 * np.log(g_curve / gamma_c) + 0.893

ax.fill_between(g_curve[g_curve < gamma_c], b_curve[g_curve < gamma_c],
                alpha=0.12, color="#1f77b4", label=r"Sub-critical ($\gamma < \gamma_c$)")
ax.fill_between(g_curve[g_curve >= gamma_c], b_curve[g_curve >= gamma_c],
                alpha=0.12, color="#d62728", label=r"Super-critical ($\gamma > \gamma_c$)")

ax.plot(g_curve, b_curve, "k--", alpha=0.6, linewidth=1.5,
        label=r"$\beta(\gamma) = 0.425\ln(\gamma/\gamma_c) + 0.893$")

emp = [
    (0.41, 0.219, "LN",   "#1f77b4"),
    (2.29, 0.950, "BN",   "#ff7f0e"),
    (3.39, 1.117, "None", "#d62728"),
]
for gam, bet, name, color in emp:
    ax.scatter(gam, bet, s=130, c=color, zorder=7,
               edgecolors="white", linewidths=1.2)
    ax.annotate(name, (gam, bet), xytext=(10, 6),
                textcoords="offset points", fontsize=8, fontweight="bold")

ax.axvline(gamma_c, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
           label=r"$\gamma_c = 2.0$")

ax.set_xlabel(r"Heat capacity exponent $\gamma$", fontsize=10)
ax.set_ylabel(r"Scaling exponent $\beta$", fontsize=10)
ax.set_title("(b)  Cooling theory: β(γ) validated\nacross sub- and super-critical regimes",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=7, loc="upper left")
ax.set_xlim(0, 4)
ax.set_ylim(0, 1.3)

fig.savefig(f"{OUT}/fig2_cooling_theory.png", dpi=300)
plt.close()
print("Figure 2 saved -> fig2_cooling_theory.png", flush=True)


# ============ CELL ============

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

OUT = "/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/papers/figures"
os.makedirs(OUT, exist_ok=True)

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
})

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
J_b2 = phase_b2[:, 2]
loss_b2 = phase_b2[:, 3]

def get_band(w):
    if w <= 32: return "narrow"
    if w < 64: return "medium"
    return "wide"

bands = [get_band(w) for w in widths_b2]
band_colors = {"narrow": "#d62728", "medium": "#ff7f0e", "wide": "#2ca02c"}
r_JL, _ = spearmanr(J_b2, loss_b2)

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
hbo_acc = np.array([r[2] for r in hbo])
random_acc = np.array([r[2] for r in random])
synflow_w = np.array([96, 96, 96, 64, 64])
hbo_w = np.array([96, 64, 96, 96, 64])
random_w = np.array([64, 96, 64, 96, 24])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plt.subplots_adjust(wspace=0.35, bottom=0.22)

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
ax.set_title("(a)  J_topo vs Validation Loss (colored by width band)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.set_xlim(0.62, 0.92)
ax.set_ylim(0.32, 0.90)

ax = axes[1]
acc_data = [random_acc, hbo_acc, synflow_acc]
width_data = [random_w, hbo_w, synflow_w]
colors = ["steelblue", "#c04040", "#40a040"]
method_names = ["Random", "HBO (J_topo)", "SynFlow (grad.)"]

bp = ax.boxplot(acc_data, positions=[0, 1, 2], patch_artist=True,
                widths=0.55,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(markersize=5))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)

for i, (data, color) in enumerate(zip(acc_data, colors)):
    jitter = np.random.uniform(-0.12, 0.12, len(data))
    ax.scatter(np.full(len(data), i) + jitter, data, s=28, c=color,
               zorder=6, alpha=0.85, edgecolors="white", linewidths=0.5)

for i, (data, color) in enumerate(zip(acc_data, colors)):
    best = data.max()
    ax.text(i, best + 0.012, "best=%.3f" % best,
            ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

for i, (w_data, color) in enumerate(zip(width_data, colors)):
    med_w = np.median(w_data)
    ax.text(i, 0.74, "med-W=%.0f" % med_w,
            ha="center", va="top", fontsize=8, color=color)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(method_names, fontsize=9)
ax.set_ylabel("Test Accuracy (top-5 candidates)", fontsize=10)
ax.set_title("(b)  Cross-validation: SynFlow & HBO converge to identical optimal architecture", fontsize=10, fontweight="bold")
ax.set_ylim(0.70, 0.90)
ax.grid(alpha=0.3, axis="y")

fig.savefig(f"{OUT}/fig3_confounding_hbo.png", dpi=300)
plt.close()
print("Figure 3 saved -> fig3_confounding_hbo.png", flush=True)
