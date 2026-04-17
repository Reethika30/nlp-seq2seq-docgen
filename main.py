"""
Automated NLP Document Generation — Main Entry Point
=====================================================
Orchestrates the full pipeline:
  1. Generate/load training data
  2. Preprocess and tokenize
  3. Build Seq2Seq model with Bahdanau Attention
  4. Train with teacher forcing
  5. Generate sample documents
"""

import os
import sys
import json
import random
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import (
    generate_synthetic_pairs, prepare_dataset,
    create_batches, save_processed_data, Vocabulary
)
from model import build_model
from train import train
from inference import generate_document


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    "num_pairs": 5000,
    "max_src_len": 150,
    "max_tgt_len": 60,
    "min_freq": 2,
    "max_vocab_size": 8000,
    "train_split": 0.85,
    "batch_size": 32,

    # Model
    "embed_dim": 256,
    "hidden_dim": 256,
    "attention_dim": 128,
    "n_layers": 2,
    "dropout": 0.3,

    # Training
    "epochs": 15,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "teacher_forcing_start": 1.0,
    "teacher_forcing_end": 0.3,
    "lr_patience": 3,
}


def main():
    print("=" * 70)
    print("AUTOMATED NLP DOCUMENT GENERATION")
    print("Seq2Seq with Bahdanau Attention")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ========================================================================
    # Step 1: Generate synthetic training data
    # ========================================================================
    print(f"\n[1/5] Generating {CONFIG['num_pairs']} synthetic document pairs...")
    raw_pairs = generate_synthetic_pairs(num_pairs=CONFIG["num_pairs"])
    print(f"  Generated {len(raw_pairs)} source-target pairs")
    print(f"  Sample source: {raw_pairs[0][0][:100]}...")
    print(f"  Sample target: {raw_pairs[0][1][:100]}...")

    # ========================================================================
    # Step 2: Preprocess and build vocabularies
    # ========================================================================
    print(f"\n[2/5] Preprocessing & tokenization...")
    encoded_pairs, src_vocab, tgt_vocab = prepare_dataset(
        raw_pairs,
        max_src_len=CONFIG["max_src_len"],
        max_tgt_len=CONFIG["max_tgt_len"],
        min_freq=CONFIG["min_freq"],
        max_vocab_size=CONFIG["max_vocab_size"],
    )

    # Save processed data
    stats = save_processed_data(encoded_pairs, src_vocab, tgt_vocab, data_dir)

    # Train/val split
    random.seed(42)
    random.shuffle(encoded_pairs)
    split_idx = int(len(encoded_pairs) * CONFIG["train_split"])
    train_pairs = encoded_pairs[:split_idx]
    val_pairs = encoded_pairs[split_idx:]

    print(f"  Train: {len(train_pairs)} | Validation: {len(val_pairs)}")

    # Create batches
    train_batches = create_batches(train_pairs, batch_size=CONFIG["batch_size"])
    val_batches = create_batches(val_pairs, batch_size=CONFIG["batch_size"])
    print(f"  Train batches: {len(train_batches)} | Val batches: {len(val_batches)}")

    # ========================================================================
    # Step 3: Build model
    # ========================================================================
    print(f"\n[3/5] Building Seq2Seq model with Bahdanau Attention...")
    model = build_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        attention_dim=CONFIG["attention_dim"],
        n_layers=CONFIG["n_layers"],
        dropout=CONFIG["dropout"],
        pad_idx=Vocabulary.PAD_IDX,
        sos_idx=Vocabulary.SOS_IDX,
        device=device,
    )

    print(f"\n  Architecture:")
    print(f"    Encoder: Bidirectional GRU, {CONFIG['n_layers']} layers, "
          f"{CONFIG['hidden_dim']} hidden")
    print(f"    Attention: Bahdanau (Additive), {CONFIG['attention_dim']} dim")
    print(f"    Decoder: GRU + Attention, {CONFIG['n_layers']} layers, "
          f"{CONFIG['hidden_dim']} hidden")
    print(f"    Embedding: {CONFIG['embed_dim']} dim")

    # ========================================================================
    # Step 4: Train
    # ========================================================================
    print(f"\n[4/5] Training...")
    history = train(model, train_batches, val_batches, CONFIG, device, model_dir)

    # ========================================================================
    # Step 5: Generate sample documents
    # ========================================================================
    print(f"\n[5/5] Generating sample documents...")

    # Load best model
    checkpoint = torch.load(os.path.join(model_dir, "best_model.pt"),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Test with sample inputs
    test_sources = [
        "The quarterly financial report for TechNova indicates revenue of $2500M, "
        "representing a 15% increase year over year. Operating expenses increased "
        "to $1200M. Net income was $450M. The board approved a dividend of $2.50 "
        "per share. Management projects continued growth in the coming quarters "
        "driven by AI integration.",

        "The ProMax X1 by CloudPeak features a 8-core processor, 6000mAh battery, "
        "and AI-powered assistant. It is designed for professionals who need high "
        "performance computing. Available in Black, Silver, and Blue, the device "
        "weighs 195g and includes fast charging and biometric auth. Pricing starts "
        "at $999.",

        "This study examines the relationship between remote work frequency and "
        "productivity using a dataset of 5000 observations from Fortune 500 "
        "companies. We employ regression analysis to analyze temporal patterns. "
        "Results indicate a strong positive correlation (p < 0.001). The findings "
        "suggest that targeted interventions improve outcomes.",
    ]

    generation_results = []
    for i, source in enumerate(test_sources, 1):
        print(f"\n  --- Sample {i} ---")
        print(f"  Source: {source[:120]}...")

        # Greedy
        greedy_text, g_meta = generate_document(
            model, source, src_vocab, tgt_vocab,
            method="greedy", max_len=60, device=device
        )
        print(f"  Greedy:  {greedy_text}")

        # Beam search
        beam_text, b_meta = generate_document(
            model, source, src_vocab, tgt_vocab,
            method="beam", beam_width=5, max_len=60, device=device
        )
        print(f"  Beam(5): {beam_text}")

        generation_results.append({
            "source": source,
            "greedy_output": greedy_text,
            "beam_output": beam_text,
            "beam_score": b_meta.get("score"),
        })

    # Save generation results
    results_path = os.path.join(output_dir, "generation_results.json")
    with open(results_path, "w") as f:
        json.dump(generation_results, f, indent=2)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"  Data:    {data_dir}")
    print(f"  Models:  {model_dir}")
    print(f"  Outputs: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
