"""
Inference & Document Generation
================================
Greedy and beam search decoding for the trained Seq2Seq model.
Generates document summaries from source text inputs.
"""

import torch
import torch.nn.functional as F


def greedy_decode(model, src_tensor, src_length, tgt_vocab,
                  max_len=100, device=None):
    """
    Greedy decoding: always pick the highest-probability token.

    Args:
        model: trained Seq2Seq model
        src_tensor: (1, src_len) source token indices
        src_length: (1,) actual source length
        tgt_vocab: target Vocabulary object
        max_len: maximum output length
        device: torch device

    Returns:
        tokens: list of decoded tokens
        attention_weights: (tgt_len, src_len) attention matrix
    """
    device = device or next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_length)
        mask = model.create_mask(src_tensor)

        input_token = torch.tensor([tgt_vocab.SOS_IDX], device=device)
        tokens = []
        attention_weights = []

        for _ in range(max_len):
            prediction, hidden, attn = model.decoder(
                input_token, hidden, encoder_outputs, mask
            )
            attention_weights.append(attn.squeeze(0).cpu())

            top_token = prediction.argmax(dim=-1).item()

            if top_token == tgt_vocab.EOS_IDX:
                break

            tokens.append(tgt_vocab.idx2word.get(top_token, tgt_vocab.UNK_TOKEN))
            input_token = torch.tensor([top_token], device=device)

    attn_matrix = torch.stack(attention_weights) if attention_weights else torch.zeros(1, 1)
    return tokens, attn_matrix


def beam_search_decode(model, src_tensor, src_length, tgt_vocab,
                       beam_width=5, max_len=100, device=None):
    """
    Beam search decoding: maintain top-k hypotheses at each step.

    Args:
        model: trained Seq2Seq model
        src_tensor: (1, src_len) source token indices
        src_length: (1,) actual source length
        tgt_vocab: target Vocabulary object
        beam_width: number of beams
        max_len: maximum output length
        device: torch device

    Returns:
        best_tokens: list of decoded tokens from best beam
        best_score: log probability of best sequence
    """
    device = device or next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_length)
        mask = model.create_mask(src_tensor)

        # Each beam: (score, token_ids, hidden_state)
        beams = [(0.0, [tgt_vocab.SOS_IDX], hidden)]
        completed = []

        for step in range(max_len):
            candidates = []

            for score, seq, h in beams:
                last_token = torch.tensor([seq[-1]], device=device)

                prediction, new_h, _ = model.decoder(
                    last_token, h, encoder_outputs, mask
                )
                log_probs = F.log_softmax(prediction, dim=-1).squeeze(0)

                # Get top-k tokens
                topk_scores, topk_indices = log_probs.topk(beam_width)

                for i in range(beam_width):
                    token_id = topk_indices[i].item()
                    new_score = score + topk_scores[i].item()
                    new_seq = seq + [token_id]

                    if token_id == tgt_vocab.EOS_IDX:
                        # Length-normalize the score
                        normalized = new_score / len(new_seq)
                        completed.append((normalized, new_seq))
                    else:
                        candidates.append((new_score, new_seq, new_h))

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        # If no completed sequences, use best beam
        if not completed:
            best = beams[0]
            completed.append((best[0] / len(best[1]), best[1]))

        # Select best completed sequence
        completed.sort(key=lambda x: x[0], reverse=True)
        best_score, best_seq = completed[0]

        # Convert to tokens (skip SOS and EOS)
        best_tokens = [
            tgt_vocab.idx2word.get(idx, tgt_vocab.UNK_TOKEN)
            for idx in best_seq
            if idx not in (tgt_vocab.SOS_IDX, tgt_vocab.EOS_IDX, tgt_vocab.PAD_IDX)
        ]

    return best_tokens, best_score


def generate_document(model, source_text, src_vocab, tgt_vocab,
                      method="greedy", beam_width=5, max_len=100, device=None):
    """
    End-to-end document generation from raw source text.

    Args:
        model: trained Seq2Seq model
        source_text: raw input string
        src_vocab: source Vocabulary
        tgt_vocab: target Vocabulary
        method: "greedy" or "beam"
        beam_width: beam size (if beam search)
        max_len: max output tokens
        device: torch device

    Returns:
        generated_text: decoded output string
        metadata: dict with attention weights or scores
    """
    from preprocessing import tokenize

    device = device or next(model.parameters()).device

    # Preprocess
    tokens = tokenize(source_text)
    indices = src_vocab.encode(tokens)

    src_tensor = torch.tensor([indices], dtype=torch.long, device=device)
    src_length = torch.tensor([len(indices)], dtype=torch.long, device=device)

    if method == "beam":
        output_tokens, score = beam_search_decode(
            model, src_tensor, src_length, tgt_vocab,
            beam_width=beam_width, max_len=max_len, device=device
        )
        metadata = {"score": score, "method": "beam", "beam_width": beam_width}
    else:
        output_tokens, attn_matrix = greedy_decode(
            model, src_tensor, src_length, tgt_vocab,
            max_len=max_len, device=device
        )
        metadata = {"attention": attn_matrix, "method": "greedy"}

    generated_text = " ".join(output_tokens)
    # Capitalize first letter and fix spacing around punctuation
    if generated_text:
        generated_text = generated_text[0].upper() + generated_text[1:]
        for punct in [".", ",", "!", "?", ";", ":"]:
            generated_text = generated_text.replace(f" {punct}", punct)

    return generated_text, metadata
