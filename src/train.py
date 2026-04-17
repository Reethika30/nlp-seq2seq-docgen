"""
Training Pipeline for Seq2Seq Document Generation
==================================================
Handles model training with:
  - Teacher forcing with scheduled decay
  - Gradient clipping
  - Learning rate scheduling
  - Epoch-level metrics tracking
  - Checkpoint saving
"""

import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_epoch(model, batches, optimizer, criterion, clip, device,
                teacher_forcing_ratio=0.5):
    """Train for a single epoch over all batches."""
    model.train()
    epoch_loss = 0
    total_tokens = 0

    for batch in batches:
        src = torch.tensor(batch["src"], dtype=torch.long, device=device)
        tgt = torch.tensor(batch["tgt"], dtype=torch.long, device=device)
        src_lengths = torch.tensor(batch["src_lengths"], dtype=torch.long, device=device)

        optimizer.zero_grad()

        # Forward pass (skip SOS in target for loss computation)
        outputs, _ = model(src, tgt[:, :-1], src_lengths, teacher_forcing_ratio)
        # outputs: (batch, tgt_len-1, vocab_size)
        # target: shift by 1 (predict next token)
        target = tgt[:, 1:]  # (batch, tgt_len-1)

        # Reshape for loss
        output_dim = outputs.size(-1)
        outputs_flat = outputs.contiguous().view(-1, output_dim)
        target_flat = target.contiguous().view(-1)

        loss = criterion(outputs_flat, target_flat)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # Track loss (exclude padding from count)
        non_pad = (target_flat != 0).sum().item()
        epoch_loss += loss.item() * non_pad
        total_tokens += non_pad

    return epoch_loss / max(total_tokens, 1)


def evaluate(model, batches, criterion, device):
    """Evaluate model on validation batches."""
    model.eval()
    epoch_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in batches:
            src = torch.tensor(batch["src"], dtype=torch.long, device=device)
            tgt = torch.tensor(batch["tgt"], dtype=torch.long, device=device)
            src_lengths = torch.tensor(batch["src_lengths"], dtype=torch.long, device=device)

            outputs, _ = model(src, tgt[:, :-1], src_lengths,
                               teacher_forcing_ratio=0)  # No teacher forcing
            target = tgt[:, 1:]

            output_dim = outputs.size(-1)
            outputs_flat = outputs.contiguous().view(-1, output_dim)
            target_flat = target.contiguous().view(-1)

            loss = criterion(outputs_flat, target_flat)

            non_pad = (target_flat != 0).sum().item()
            epoch_loss += loss.item() * non_pad
            total_tokens += non_pad

    return epoch_loss / max(total_tokens, 1)


def train(model, train_batches, val_batches, config, device, save_dir):
    """
    Full training loop with scheduling and checkpointing.

    Args:
        model: Seq2Seq model
        train_batches: list of training batch dicts
        val_batches: list of validation batch dicts
        config: dict with hyperparameters
        device: torch device
        save_dir: directory to save checkpoints and logs
    """
    os.makedirs(save_dir, exist_ok=True)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"],
                           weight_decay=config.get("weight_decay", 0))
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                   patience=config.get("lr_patience", 3))

    # Teacher forcing schedule
    tf_start = config.get("teacher_forcing_start", 1.0)
    tf_end = config.get("teacher_forcing_end", 0.3)
    tf_decay = (tf_start - tf_end) / max(config["epochs"], 1)

    best_val_loss = float("inf")
    history = []

    print(f"\n  Training for {config['epochs']} epochs...")
    print(f"  Train batches: {len(train_batches)} | Val batches: {len(val_batches)}")
    print(f"  Device: {device}")
    print("-" * 70)

    for epoch in range(1, config["epochs"] + 1):
        tf_ratio = max(tf_end, tf_start - tf_decay * epoch)
        start_time = time.time()

        train_loss = train_epoch(
            model, train_batches, optimizer, criterion,
            clip=config.get("grad_clip", 1.0),
            device=device,
            teacher_forcing_ratio=tf_ratio
        )

        val_loss = evaluate(model, val_batches, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        train_ppl = math.exp(min(train_loss, 100))
        val_ppl = math.exp(min(val_loss, 100))
        lr = optimizer.param_groups[0]["lr"]

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_ppl": round(train_ppl, 2),
            "val_ppl": round(val_ppl, 2),
            "tf_ratio": round(tf_ratio, 3),
            "lr": lr,
            "time_sec": round(elapsed, 1),
        }
        history.append(epoch_data)

        print(f"  Epoch {epoch:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} PPL: {train_ppl:8.2f} | "
              f"Val Loss: {val_loss:.4f} PPL: {val_ppl:8.2f} | "
              f"TF: {tf_ratio:.2f} LR: {lr:.6f} | {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, os.path.join(save_dir, "best_model.pt"))
            print(f"    -> Saved best model (val_loss: {val_loss:.4f})")

    # Save training history
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save final model
    torch.save({
        "epoch": config["epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": config,
    }, os.path.join(save_dir, "final_model.pt"))

    print("-" * 70)
    print(f"  Training complete. Best val loss: {best_val_loss:.4f} "
          f"(PPL: {math.exp(min(best_val_loss, 100)):.2f})")
    print(f"  Models saved to: {save_dir}")

    return history
