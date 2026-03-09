import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} results.csv")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: CPU vs GPU time (block_size=16, log-log)
d16 = df[df["block_size"] == 16]
ax = axes[0]
ax.plot(d16["N"], d16["cpu_ms"], "o-", label="CPU")
ax.plot(d16["N"], d16["gpu_ms"], "s-", label="GPU (block=16)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Matrix size N")
ax.set_ylabel("Time (ms)")
ax.set_title("CPU vs GPU Time")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)

# Plot 2: Speedup vs N
ax = axes[1]
ax.plot(d16["N"], d16["cpu_ms"] / d16["gpu_ms"], "o-", color="green")
ax.axhline(y=1, color="gray", ls="--", alpha=0.5)
ax.set_xscale("log", base=2)
ax.set_xlabel("Matrix size N")
ax.set_ylabel("Speedup (CPU / GPU)")
ax.set_title("GPU Speedup")
ax.grid(True, which="both", ls="--", alpha=0.5)

# Plot 3: Block size comparison
ax = axes[2]
for bs, group in df.groupby("block_size"):
    ax.plot(group["N"], group["gpu_ms"], "o-", label=f"block={bs}")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Matrix size N")
ax.set_ylabel("GPU Time (ms)")
ax.set_title("Effect of Block Size")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("Saved results.png")
