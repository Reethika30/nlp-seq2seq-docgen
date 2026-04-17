"""
Gradio Web Demo for Seq2Seq Document Generation
================================================
Interactive UI: paste a long source document, get a generated summary
with greedy or beam search decoding, plus an attention heatmap.

Run locally:    python app.py
Deploy:         HuggingFace Spaces (Gradio SDK)
"""

import os
import sys
import io
import base64
import torch
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model import build_model
from inference import generate_document, greedy_decode
from preprocessing import Vocabulary, tokenize

# ----------------------------------------------------------------------
# Load model once at startup
# ----------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")

print("Loading vocabularies...")
SRC_VOCAB = Vocabulary.load(os.path.join(BASE, "data", "src_vocab.pkl"))
TGT_VOCAB = Vocabulary.load(os.path.join(BASE, "data", "tgt_vocab.pkl"))

print("Loading model...")
MODEL = build_model(
    src_vocab_size=len(SRC_VOCAB),
    tgt_vocab_size=len(TGT_VOCAB),
    embed_dim=256, hidden_dim=256, attention_dim=128,
    n_layers=2, dropout=0.3,
    pad_idx=Vocabulary.PAD_IDX, sos_idx=Vocabulary.SOS_IDX,
    device=DEVICE,
)
ckpt = torch.load(os.path.join(BASE, "models", "best_model.pt"),
                  map_location=DEVICE, weights_only=False)
MODEL.load_state_dict(ckpt["model_state_dict"])
MODEL.eval()
print(f"Loaded checkpoint from epoch {ckpt.get('epoch')} "
      f"(val_loss: {ckpt.get('val_loss'):.4f})")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def attention_heatmap(source_text, output_tokens, attn_matrix):
    """Render attention weights as a matplotlib figure."""
    src_tokens = tokenize(source_text)[:attn_matrix.shape[1] - 2]  # trim special
    out_tokens = output_tokens[:attn_matrix.shape[0]]

    # Trim attention matrix to match displayed tokens
    attn = attn_matrix[:len(out_tokens), :len(src_tokens) + 2]

    fig, ax = plt.subplots(figsize=(max(8, len(src_tokens) * 0.25),
                                     max(4, len(out_tokens) * 0.35)))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(src_tokens) + 2))
    ax.set_xticklabels(["<SOS>"] + src_tokens + ["<EOS>"],
                       rotation=75, fontsize=8)
    ax.set_yticks(range(len(out_tokens)))
    ax.set_yticklabels(out_tokens, fontsize=9)
    ax.set_xlabel("Source Tokens")
    ax.set_ylabel("Generated Tokens")
    ax.set_title("Bahdanau Attention Weights")
    plt.colorbar(im, ax=ax, fraction=0.025)
    plt.tight_layout()
    return fig


def generate(source_text, method, beam_width, max_len):
    """Main inference function called by Gradio."""
    if not source_text.strip():
        return "Please enter source text.", None, ""

    try:
        if method == "Greedy":
            text, meta = generate_document(
                MODEL, source_text, SRC_VOCAB, TGT_VOCAB,
                method="greedy", max_len=int(max_len), device=DEVICE
            )
            # Build attention heatmap
            fig = None
            attn = meta.get("attention")
            if attn is not None and hasattr(attn, "shape"):
                out_tokens = text.split()
                try:
                    fig = attention_heatmap(source_text, out_tokens,
                                            attn.cpu().numpy() if torch.is_tensor(attn) else np.asarray(attn))
                except Exception as e:
                    print(f"Heatmap error: {e}")
            info = f"Method: Greedy decode | Output length: {len(text.split())} tokens"
            return text, fig, info

        else:  # Beam
            text, meta = generate_document(
                MODEL, source_text, SRC_VOCAB, TGT_VOCAB,
                method="beam", beam_width=int(beam_width),
                max_len=int(max_len), device=DEVICE
            )
            info = (f"Method: Beam Search (width={int(beam_width)}) | "
                    f"Score: {meta.get('score', 0):.4f} | "
                    f"Output length: {len(text.split())} tokens")
            return text, None, info

    except Exception as e:
        return f"Error: {e}", None, ""


# ----------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------
EXAMPLES = [
    ["The quarterly financial report for TechNova indicates revenue of $2500M, "
     "representing a 15% increase year over year. Operating expenses increased "
     "to $1200M. Net income was $450M. The board approved a dividend of $2.50 "
     "per share. Management projects continued growth in the coming quarters "
     "driven by AI integration.", "Greedy", 5, 60],

    ["The ProMax X1 by CloudPeak features a 8-core processor, 6000mAh battery, "
     "and AI-powered assistant. It is designed for professionals who need high "
     "performance computing. Available in Black, Silver, and Blue, the device "
     "weighs 195g and includes fast charging and biometric auth. Pricing starts "
     "at $999.", "Beam", 5, 60],

    ["This study examines the relationship between remote work frequency and "
     "productivity using a dataset of 5000 observations from Fortune 500 "
     "companies. We employ regression analysis to analyze temporal patterns. "
     "Results indicate a strong positive correlation (p < 0.001). The findings "
     "suggest that targeted interventions improve outcomes.", "Greedy", 5, 60],
]

DESCRIPTION = """
# Seq2Seq Document Generation with Bahdanau Attention

Encoder-decoder model that compresses long-form documents (financial reports,
product specs, research abstracts) into concise summaries.

- **Architecture:** Bidirectional GRU Encoder + Bahdanau (Additive) Attention + GRU Decoder
- **Parameters:** 3.9M | **Framework:** PyTorch
- **Training:** 15 epochs on 5,000 synthetic pairs | **Best Val PPL:** 9.08
- **Decoding:** Greedy + Beam Search (width 5)

Paste a document below, pick a decoding strategy, and see the model
generate a summary. Greedy mode also renders the **attention heatmap**
showing which source tokens the decoder focused on at each step.
"""

with gr.Blocks(title="Seq2Seq Doc Generation") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            src = gr.Textbox(
                label="Source Document",
                placeholder="Paste a long document here...",
                lines=8,
            )
            with gr.Row():
                method = gr.Radio(
                    ["Greedy", "Beam"], value="Greedy",
                    label="Decoding Method"
                )
                beam_width = gr.Slider(2, 10, value=5, step=1,
                                       label="Beam Width")
                max_len = gr.Slider(20, 120, value=60, step=5,
                                    label="Max Output Tokens")
            btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(label="Generated Summary", lines=4)
            info = gr.Textbox(label="Decoding Info", lines=1)
            heatmap = gr.Plot(label="Attention Heatmap (Greedy only)")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[src, method, beam_width, max_len],
        outputs=[output, heatmap, info],
        fn=generate,
        cache_examples=False,
    )

    btn.click(generate,
              inputs=[src, method, beam_width, max_len],
              outputs=[output, heatmap, info])

    gr.Markdown(
        "---\n"
        "**Repo:** [github.com/Reethika30/nlp-seq2seq-docgen]"
        "(https://github.com/Reethika30/nlp-seq2seq-docgen)"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
