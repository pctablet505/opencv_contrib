# Draws the two figures from the measurement logs. Run: python plot_figures.py <logs_dir> <out_dir>
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

logs, out = sys.argv[1], sys.argv[2]
RED, BLUE, GREY, INK, MUTED = "#c62828", "#1f5fbf", "#6b6b6b", "#1f2328", "#59636e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#d0d7de",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def rows(name):
    with open(f"{logs}/{name}") as f:
        return [json.loads(l) for l in f if l.startswith("{")]


# 1. normalization error against input mean / std
stock = [r for r in rows("stock-norm_offset.log") if r.get("layer") == "LayerNormalization"]
fixed = [r for r in rows("pr-norm_offset.log") if r.get("layer") == "LayerNormalization"]
mvn = {(r["backend"], r["mean"]): r["max_err"] for r in rows("stock-norm_offset.log") if r.get("layer") == "MVN"}
assert all(abs(mvn[(r["backend"], r["mean"])] - r["max_err"]) < 1e-9 for r in stock), "MVN and LayerNorm differ"

def series(rs, backend):
    s = [r for r in rs if r["backend"] == backend]
    return [r["mean"] / r["std"] for r in s], [max(r["max_err"], 1e-7) for r in s], [r["nan"] / r["total"] for r in s]

def norm_chart(with_fix, path):
    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=150)
    x, y, nan = series(stock, "CUDA")
    ok = [i for i, f in enumerate(nan) if f == 0]
    bad = [i for i, f in enumerate(nan) if f > 0]
    NAN_Y = 3e3  # points with NaN outputs sit on a band at the top instead of at their largest finite error
    yb = [NAN_Y if nan[i] > 0 else y[i] for i in range(len(y))]
    ax.axhspan(1.2e3, 4e4, color="#fbe9e7", zorder=0)
    ax.plot(x, yb, color=RED, lw=1.8, zorder=3)
    ax.plot([x[i] for i in ok], [y[i] for i in ok], "o", color=RED, ms=5, zorder=4, label="CUDA, 5.x")
    ax.plot([x[i] for i in bad], [NAN_Y] * len(bad), "X", color=RED, ms=8, zorder=4, label="CUDA, 5.x: some outputs NaN")
    xc, yc, _ = series(stock, "CPU")
    ax.plot(xc, yc, "s-", color=GREY, lw=1.4, ms=4, zorder=2, label="CPU, 5.x (finite but wrong)")
    if with_fix:
        xf, yf, _ = series(fixed, "CUDA")
        ax.plot(xf, yf, "o-", color=BLUE, lw=1.8, ms=5, zorder=3, label="CUDA with this fix")
    ax.axhline(1.73, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(x[0], 2.3, "correct outputs lie within ±1.73", color=MUTED, fontsize=9, va="bottom")
    lo, hi = min(nan[i] for i in bad), max(nan[i] for i in bad)
    ax.text(x[bad[0]] * 1.1, 1.25e4, f"NaN in {lo:.0%}–{hi:.0%} of the outputs", color=RED, fontsize=9.5, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-7, 4e4)
    ax.set_xlabel("input mean ÷ standard deviation (64 rows of 768 values, std 0.29)")
    ax.set_ylabel("largest |output error| vs FP64")
    ax.grid(True, which="major", color="#eaeef2", lw=0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path)


norm_chart(True, f"{out}/norm-error-by-offset.png")
norm_chart(False, f"{out}/norm-error-by-offset-5x.png")

# 2. Attention device memory against sequence length
fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=150)
st, pr = rows("stock-attention_sweep.log"), rows("pr-attention_sweep.log")
for prec, ls in (("fp32", "-"), ("fp16", (0, (5, 3)))):
    for data, color, who in ((st, RED, "5.x"), (pr, BLUE, "with this PR")):
        d = [r for r in data if r["precision"] == prec]
        xs, ys = [r["S"] for r in d], [r["device_mib"] for r in d]
        ax.plot(xs, ys, ls=ls, color=color, lw=1.8, marker="o", ms=4.5, label=f"{prec.upper()}, {who}")
        ax.annotate(f"{ys[-1]:.0f} MiB", xy=(xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points",
                    color=INK, fontsize=9.5, va="center")
ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(FixedLocator([256, 512, 1024, 2048, 4096]))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
ax.set_xlim(220, 6200)
ax.set_xlabel("sequence length S  (batch 1, 12 heads, hidden 768)")
ax.set_ylabel("device memory kept by the node (MiB)")
ax.grid(True, axis="y", color="#eaeef2", lw=0.8)
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{out}/attention-memory-by-length.png")
print("ok")
