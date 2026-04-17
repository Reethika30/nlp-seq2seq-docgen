"""
Standalone Inference: Loads the trained checkpoint and generates documents.
Runs Step 5 of the pipeline independently.
"""

import os
import sys
import json
import pickle
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model import build_model
from inference import generate_document
from preprocessing import Vocabulary


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # Load vocabularies
    print("Loading vocabularies...")
    src_vocab = Vocabulary.load(os.path.join(data_dir, "src_vocab.pkl"))
    tgt_vocab = Vocabulary.load(os.path.join(data_dir, "tgt_vocab.pkl"))
    print(f"  Source vocab: {len(src_vocab)} | Target vocab: {len(tgt_vocab)}")

    # Build model
    print("Building model...")
    model = build_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=256,
        hidden_dim=256,
        attention_dim=128,
        n_layers=2,
        dropout=0.3,
        pad_idx=Vocabulary.PAD_IDX,
        sos_idx=Vocabulary.SOS_IDX,
        device=device,
    )

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt_path = os.path.join(model_dir, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}, "
          f"val_loss: {checkpoint.get('val_loss', 'N/A')}")

    # Test sources
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
    print("\n" + "=" * 70)
    print("GENERATING DOCUMENTS")
    print("=" * 70)

    for i, source in enumerate(test_sources, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Source: {source[:120]}...")

        greedy_text, g_meta = generate_document(
            model, source, src_vocab, tgt_vocab,
            method="greedy", max_len=60, device=device
        )
        print(f"Greedy:  {greedy_text}")

        beam_text, b_meta = generate_document(
            model, source, src_vocab, tgt_vocab,
            method="beam", beam_width=5, max_len=60, device=device
        )
        print(f"Beam(5): {beam_text}  (score: {b_meta.get('score'):.4f})")

        generation_results.append({
            "sample_id": i,
            "source": source,
            "greedy_output": greedy_text,
            "beam_output": beam_text,
            "beam_score": float(b_meta.get("score", 0.0)),
        })

    # Save results
    results_path = os.path.join(output_dir, "generation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(generation_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save config summary
    config = {
        "num_pairs": 5000,
        "embed_dim": 256,
        "hidden_dim": 256,
        "attention_dim": 128,
        "n_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", 0.0)),
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
