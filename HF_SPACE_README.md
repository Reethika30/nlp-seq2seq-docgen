---
title: Seq2Seq Document Generation
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Seq2Seq Document Generation with Bahdanau Attention

Interactive demo of an encoder-decoder model that generates concise summaries from long-form documents.

- **Architecture:** Bidirectional GRU Encoder + Bahdanau (Additive) Attention + GRU Decoder
- **Parameters:** 3.9M | **Framework:** PyTorch
- **Training:** 15 epochs on 5,000 synthetic pairs | **Best Val PPL:** 9.08
- **Decoding:** Greedy + Beam Search with attention visualization

Source code: https://github.com/Reethika30/nlp-seq2seq-docgen
