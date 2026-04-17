"""Generate training curves plot from training_history.json."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base, "models", "training_history.json")) as f:
    hist = json.load(f)

epochs = [h["epoch"] for h in hist]
train_loss = [h["train_loss"] for h in hist]
val_loss = [h["val_loss"] for h in hist]
val_ppl = [h["val_ppl"] for h in hist]
tf_ratio = [h["tf_ratio"] for h in hist]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Loss curves
axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4", linewidth=2)
axes[0].plot(epochs, val_loss, "s-", label="Val Loss", color="#d62728", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Perplexity
axes[1].plot(epochs, val_ppl, "o-", color="#2ca02c", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation Perplexity")
axes[1].set_title("Validation Perplexity (lower = better)")
axes[1].set_yscale("log")
axes[1].grid(alpha=0.3, which="both")

# Teacher forcing schedule
axes[2].plot(epochs, tf_ratio, "o-", color="#ff7f0e", linewidth=2)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Teacher Forcing Ratio")
axes[2].set_title("Teacher Forcing Schedule (1.0 -> 0.3)")
axes[2].grid(alpha=0.3)

best_epoch = min(range(len(val_loss)), key=lambda i: val_loss[i])
fig.suptitle(
    f"Seq2Seq Training | Best Val Loss: {val_loss[best_epoch]:.4f} "
    f"(PPL {val_ppl[best_epoch]:.2f}) at Epoch {epochs[best_epoch]}",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()

out_path = os.path.join(base, "assets", "loss_curves.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Saved: {out_path}")
