"""
Seq2Seq Model with Bahdanau (Additive) Attention
=================================================
Encoder-Decoder architecture for automated document generation.

Components:
  - Encoder: Bidirectional GRU with embedding layer
  - Bahdanau Attention: Additive attention mechanism
  - Decoder: GRU with attention-augmented context vector
  - Seq2Seq: Full model wrapping encoder + decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Bidirectional GRU Encoder.

    Converts source token indices into a sequence of hidden states.
    Uses bidirectional GRU and projects the final hidden state to
    match the decoder's hidden size.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # Project bidirectional hidden to decoder hidden size
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lengths=None):
        """
        Args:
            src: (batch, src_len) source token indices
            src_lengths: (batch,) actual lengths for packing

        Returns:
            outputs: (batch, src_len, hidden*2) encoder outputs
            hidden: (n_layers, batch, hidden) final hidden state
        """
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, embed)

        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)

        # hidden: (n_layers * 2, batch, hidden) -> combine directions
        # Reshape: (n_layers, 2, batch, hidden)
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        # Concatenate forward and backward: (n_layers, batch, hidden * 2)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        # Project to decoder size: (n_layers, batch, hidden)
        hidden = torch.tanh(self.fc_hidden(hidden))

        return outputs, hidden


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.

    Computes alignment scores between the decoder hidden state and
    each encoder output using a learned additive function:

        score(s_t, h_j) = v^T * tanh(W_s * s_t + W_h * h_j)

    Returns attention weights and weighted context vector.
    """

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super().__init__()
        # encoder outputs are bidirectional, so dim = encoder_hidden * 2
        self.W_encoder = nn.Linear(encoder_hidden_dim * 2, attention_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch, decoder_hidden)
            encoder_outputs: (batch, src_len, encoder_hidden*2)
            mask: (batch, src_len) boolean mask, True = ignore

        Returns:
            context: (batch, encoder_hidden*2) weighted sum
            attention_weights: (batch, src_len) alignment scores
        """
        # Project encoder outputs: (batch, src_len, attn_dim)
        encoder_proj = self.W_encoder(encoder_outputs)

        # Project decoder hidden: (batch, attn_dim) -> (batch, 1, attn_dim)
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)

        # Additive score: (batch, src_len, attn_dim) -> (batch, src_len, 1)
        energy = torch.tanh(encoder_proj + decoder_proj)
        scores = self.v(energy).squeeze(-1)  # (batch, src_len)

        # Apply mask (set padded positions to -inf before softmax)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)  # (batch, src_len)

        # Weighted sum of encoder outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, src_len)
            encoder_outputs                   # (batch, src_len, enc_hidden*2)
        ).squeeze(1)  # (batch, enc_hidden*2)

        return context, attention_weights


class Decoder(nn.Module):
    """
    GRU Decoder with Bahdanau Attention.

    At each time step:
    1. Embed the previous output token
    2. Compute attention over encoder outputs
    3. Concatenate embedding + context vector
    4. Pass through GRU
    5. Predict next token
    """

    def __init__(self, vocab_size, embed_dim, encoder_hidden_dim,
                 decoder_hidden_dim, attention_dim, n_layers=1,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(
            encoder_hidden_dim, decoder_hidden_dim, attention_dim
        )
        self.rnn = nn.GRU(
            embed_dim + encoder_hidden_dim * 2,  # input = embed + context
            decoder_hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # Output projection: context + hidden + embedding -> vocab
        self.fc_out = nn.Linear(
            encoder_hidden_dim * 2 + decoder_hidden_dim + embed_dim,
            vocab_size
        )

    def forward(self, input_token, hidden, encoder_outputs, mask=None):
        """
        Single decoding step.

        Args:
            input_token: (batch,) previous token indices
            hidden: (n_layers, batch, dec_hidden)
            encoder_outputs: (batch, src_len, enc_hidden*2)
            mask: (batch, src_len)

        Returns:
            prediction: (batch, vocab_size) logits
            hidden: (n_layers, batch, dec_hidden) updated hidden
            attention_weights: (batch, src_len)
        """
        input_token = input_token.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1, embed)

        # Use top layer hidden for attention
        top_hidden = hidden[-1]  # (batch, dec_hidden)

        # Compute attention
        context, attn_weights = self.attention(
            top_hidden, encoder_outputs, mask
        )  # context: (batch, enc_hidden*2)

        # Concatenate embedding and context for GRU input
        rnn_input = torch.cat([
            embedded, context.unsqueeze(1)
        ], dim=-1)  # (batch, 1, embed + enc_hidden*2)

        output, hidden = self.rnn(rnn_input, hidden)
        # output: (batch, 1, dec_hidden)

        output = output.squeeze(1)  # (batch, dec_hidden)

        # Predict: concat context + output + embedded
        prediction = self.fc_out(torch.cat([
            output, context, embedded.squeeze(1)
        ], dim=-1))  # (batch, vocab_size)

        return prediction, hidden, attn_weights


class Seq2Seq(nn.Module):
    """
    Full Sequence-to-Sequence model combining Encoder and Decoder.
    Supports teacher forcing during training.
    """

    def __init__(self, encoder, decoder, pad_idx=0, sos_idx=1, device=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.device = device or torch.device("cpu")

    def create_mask(self, src):
        """Create mask for padded positions: True = ignore."""
        return src == self.pad_idx

    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch, src_len)
            tgt: (batch, tgt_len)
            src_lengths: (batch,)
            teacher_forcing_ratio: probability of using ground truth as next input

        Returns:
            outputs: (batch, tgt_len, vocab_size) logits
            attentions: (batch, tgt_len, src_len) attention weights
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size

        # Encode
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        # Create source mask
        mask = self.create_mask(src)

        # Prepare output tensors
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)
        attentions = torch.zeros(batch_size, tgt_len, src.size(1), device=self.device)

        # First input is SOS token
        input_token = torch.full(
            (batch_size,), self.sos_idx, dtype=torch.long, device=self.device
        )

        for t in range(tgt_len):
            prediction, hidden, attn_weights = self.decoder(
                input_token, hidden, encoder_outputs, mask
            )

            outputs[:, t] = prediction
            attentions[:, t, :attn_weights.size(1)] = attn_weights

            # Teacher forcing: use ground truth or predicted token
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t]
            else:
                input_token = prediction.argmax(dim=-1)

        return outputs, attentions


def build_model(src_vocab_size, tgt_vocab_size, embed_dim=256,
                hidden_dim=512, attention_dim=256, n_layers=2,
                dropout=0.3, pad_idx=0, sos_idx=1, device=None):
    """Factory function to construct the full Seq2Seq model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(
        vocab_size=src_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        pad_idx=pad_idx,
    )

    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        encoder_hidden_dim=hidden_dim,
        decoder_hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        n_layers=n_layers,
        dropout=dropout,
        pad_idx=pad_idx,
    )

    model = Seq2Seq(encoder, decoder, pad_idx=pad_idx,
                     sos_idx=sos_idx, device=device)
    model = model.to(device)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model
