#!/usr/bin/env python3
"""ThermoRG-NN paper figures. Run: python3 papers/generate_figures.py"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

OUT = "/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/papers/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": False,
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================================
# FIGURE 1 — D-scaling law: (a) loss vs width, (b) residuals, (c) R²
# ============================================================================

def plaw(D, a, b, c):
    return a * D**(-b) + c

configs = {
    "LN":   {"D": np.array([32, 48, 64, 96]), "loss": np.array([0.936, 0.857, 0.804, 0.736]),
             "std": np.array([0.021, 0.018, 0.015, 0.012]), "color": "#1f77b4"},
    "BN":   {"D": np.array([32, 48, 64, 96]), "loss": np.array([0.958, 0.800, 0.721, 0.639]),
             "std": np.array([0.021, 0.018, 0.015, 0.012]), "color": "#ff7f0e"},
    "None": {"D": np.array([32, 48, 64, 96]), "loss": np.array([1.285, 1.100, 1.011, 0.926]),
             "std": np.array([0.045, 0.038, 0.031, 0.025]), "color": "#d62728"},
}

fits = {}
for name, cfg in configs.items():
    popt, _ = curve_fit(plaw, cfg["D"], cfg["loss"],
                        p0=[10, 0.5, 0.5],
                        bounds=([0, 0, 0], [1000, 3, 3]), maxfev=10000)
    ss_res = ((cfg["loss"] - plaw(cfg["D"], *popt))**2).sum()
    ss_tot = ((cfg["loss"] - cfg["loss"].mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    fits[name] = {"popt": popt, "r2": r2}
    print(f"  {name}: beta={popt[1]:.3f}, R2={r2:.3f}", flush=True)

D_fit = np.linspace(25, 120, 200)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
fig.subplots_adjust(wspace=0.35, bottom=0.22, top=0.88)

# ---- Panel (a): D-scaling ----
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
ax.set_title("(a) D-scaling: loss vs width", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.set_xticks([32, 48, 64, 96])
ax.set_xticklabels(["32", "48", "64", "96"])
ax.set_xticks([], minor=True)
ax.set_xlim(22, 130)

# Y-axis cleanup: force plain decimal format
y_ticks = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{y:.1f}" for y in y_ticks])
ax.set_yticks([], minor=True)

ax.tick_params(labelbottom=True)

# ---- Panel (b): Residuals ----
ax = axes[1]
for name, cfg in configs.items():
    popt = fits[name]["popt"]
    residuals = cfg["loss"] - plaw(cfg["D"], *popt)
    ax.scatter(cfg["D"], residuals, color=cfg["color"], s=60, zorder=5,
               edgecolors="white", linewidths=0.7, label=name)

ax.set_xscale("log")
ax.set_xlabel("Width D", fontsize=10)
ax.set_ylabel("Residual (data − fit)", fontsize=10)
ax.set_title("(b) Residuals", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.set_xticks([32, 48, 64, 96])
ax.set_xticklabels(["32", "48", "64", "96"])
ax.set_xticks([], minor=True)  # clear auto minor ticks to avoid double labels
ax.set_xlim(22, 130)
ax.tick_params(labelbottom=True)
ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)

# ---- Panel (c): R² bar chart ----
ax = axes[2]
names = list(configs.keys())
r2_vals = [fits[n]["r2"] for n in names]
colors_b = [configs[n]["color"] for n in names]
bars = ax.bar(names, r2_vals, color=colors_b, width=0.5, edgecolor="white", linewidth=0.7)

ax.set_ylabel("R² goodness of fit", fontsize=10)
ax.set_title("(c) R²", fontsize=10, fontweight="bold")
ax.set_ylim(0.98, 1.002)
ax.axhline(0.99, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)

fig.savefig(f"{OUT}/fig1_D_scaling_Jtopo.png", dpi=300)
plt.close()
print("Figure 1 saved", flush=True)

# ============================================================================
# FIGURE 2 — Cooling theory: (a) β vs γ, (b) predicted vs actual, (c) width trajectory
# ============================================================================

gamma_c = 2.0
fig2_data = {
    "LN":   {"gamma": 0.41, "beta": 0.219, "color": "#1f77b4"},
    "BN":   {"gamma": 2.29, "beta": 0.950, "color": "#ff7f0e"},
    "None": {"gamma": 3.39, "beta": 1.117, "color": "#d62728"},
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
fig.subplots_adjust(wspace=0.35, bottom=0.22, top=0.88)

# Panel (a): β vs γ
ax = axes[0]
g_curve = np.linspace(0.3, 4.0, 300)
b_curve = 0.425 * np.log(g_curve / gamma_c) + 0.893

ax.fill_between(g_curve[g_curve < gamma_c], b_curve[g_curve < gamma_c],
                alpha=0.12, color="#1f77b4")
ax.fill_between(g_curve[g_curve >= gamma_c], b_curve[g_curve >= gamma_c],
                alpha=0.12, color="#d62728")
ax.plot(g_curve, b_curve, "k--", alpha=0.6, linewidth=1.5,
        label=r"$\beta(\gamma) = 0.425\ln(\gamma/\gamma_c)+0.893$")

for name, info in fig2_data.items():
    ax.scatter(info["gamma"], info["beta"], s=130, c=info["color"], zorder=7,
               edgecolors="white", linewidths=1.2)
    ax.annotate(name, (info["gamma"], info["beta"]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=9, fontweight="bold")

ax.axvline(gamma_c, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
           label=r"$\gamma_c=2.0$")
ax.set_xlabel(r"Heat capacity exponent $\gamma$", fontsize=10)
ax.set_ylabel(r"Scaling exponent $\beta$", fontsize=10)
ax.set_title("(a)  β vs γ: cooling theory", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper left")
ax.set_xlim(0, 4)
ax.set_ylim(-0.1, 1.3)
ax.tick_params(labelbottom=True)

# Panel (b): Predicted vs actual β
ax = axes[1]
gammas = [v["gamma"] for v in fig2_data.values()]
betas  = [v["beta"]  for v in fig2_data.values()]
colors_b = [v["color"] for v in fig2_data.values()]
names_b  = list(fig2_data.keys())
b_pred = [0.425 * np.log(g / gamma_c) + 0.893 for g in gammas]

for g, b, bp, c, n in zip(gammas, betas, b_pred, colors_b, names_b):
    err = abs(bp - b) / b * 100
    ax.scatter(b, bp, s=130, c=c, zorder=7, edgecolors="white", linewidths=1.2)
    ax.annotate(f"{n}\n({err:.1f}% err)", (b, bp),
                xytext=(8, 0), textcoords="offset points",
                fontsize=8, fontweight="bold")

lims = [0, 1.3]
ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1.2, label="identity")
ax.set_xlabel(r"Actual $\beta$", fontsize=10)
ax.set_ylabel(r"Predicted $\beta$", fontsize=10)
ax.set_title("(b)  Predicted vs actual β", fontsize=10, fontweight="bold")
ax.set_xlim(0, 1.3)
ax.set_ylim(0, 1.3)
ax.set_aspect("equal")
ax.legend(fontsize=8)
ax.tick_params(labelbottom=True)

# Panel (c): Width-scaling trajectories
ax = axes[2]
D_traj = np.linspace(20, 120, 200)
for name, cfg in configs.items():
    popt = fits[name]["popt"]
    ax.plot(D_traj, plaw(D_traj, *popt), "-", color=cfg["color"],
            linewidth=1.8, label=name, alpha=0.8)
    ax.scatter(cfg["D"], cfg["loss"], color=cfg["color"], s=50, zorder=5,
               edgecolors="white", linewidths=0.7)

ax.set_xlim(20, 120)
ax.set_xlabel("Width D", fontsize=10)
ax.set_ylabel("Loss", fontsize=10)
ax.set_title("(c)  Width-scaling trajectories", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.set_xticks([32, 48, 64, 96])
ax.set_xticklabels(["32", "48", "64", "96"])
ax.tick_params(labelbottom=True)

fig.savefig(f"{OUT}/fig2_cooling_theory.png", dpi=300)
plt.close()
print("Figure 2 saved", flush=True)

# ============================================================================
# FIGURE 3 — Confounding + cross-validation: (a) J_topo vs loss, (b) boxplot
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
    return "narrow" if w <= 32 else ("medium" if w < 64 else "wide")

bands = [get_band(w) for w in widths_b2]
band_colors = {"narrow": "#d62728", "medium": "#ff7f0e", "wide": "#2ca02c"}

synflow_acc = np.array([0.8724, 0.8587, 0.8607, 0.8398, 0.7631])
hbo_acc     = np.array([0.8744, 0.8486, 0.8514, 0.8440, 0.8277])
random_acc  = np.array([0.8515, 0.8477, 0.8149, 0.8157, 0.7700])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.subplots_adjust(wspace=0.35, bottom=0.22)

# Panel (a): J_topo vs loss, colored by band
ax = axes[0]
for band_name in ["narrow", "medium", "wide"]:
    mask = np.array([b == band_name for b in bands])
    ax.scatter(J_b2[mask], loss_b2[mask],
               c=band_colors[band_name], s=90, label=band_name.title(),
               zorder=5, edgecolors="white", linewidths=0.7)
    if sum(mask) >= 2:
        z = np.polyfit(J_b2[mask], loss_b2[mask], 1)
        J_rng = np.linspace(J_b2[mask].min(), J_b2[mask].max(), 50)
        ax.plot(J_rng, np.poly1d(z)(J_rng), "--", color=band_colors[band_name], alpha=0.5, linewidth=1.5)

ax.set_xlabel("J_topo", fontsize=11)
ax.set_ylabel("Validation loss", fontsize=11)
ax.set_title("(a)  J_topo vs Validation Loss", fontsize=10, fontweight="bold")
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.set_xlim(0.62, 0.92)
ax.set_ylim(0.32, 0.90)
ax.tick_params(labelbottom=True)

# Panel (b): Cross-validation boxplot
ax = axes[1]
acc_data    = [random_acc, hbo_acc, synflow_acc]
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
    ax.text(i, best + 0.012, f"best={best:.3f}",
            ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(method_names, fontsize=9)
ax.set_ylabel("Test Accuracy (top-5 candidates)", fontsize=10)
ax.set_title("(b)  Cross-validation: SynFlow vs ThermoRG-AL", fontsize=10, fontweight="bold")
ax.set_ylim(0.70, 0.90)
ax.grid(alpha=0.3, axis="y")
ax.tick_params(labelbottom=True)

fig.savefig(f"{OUT}/fig3_confounding_hbo.png", dpi=300)
plt.close()
print("Figure 3 saved", flush=True)

print("\nAll done.", flush=True)