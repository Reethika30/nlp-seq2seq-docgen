"""
Data Preprocessing Pipeline for Seq2Seq Document Generation
============================================================
Handles ingestion, cleaning, tokenization, and batching of
unstructured text data for training an encoder-decoder model
with Bahdanau Attention.
"""

import re
import os
import json
import pickle
import unicodedata
from collections import Counter


# ============================================================================
# VOCABULARY
# ============================================================================

class Vocabulary:
    """Word-level vocabulary with special tokens for Seq2Seq training."""

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, min_freq=2, max_vocab_size=None):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()
        self._built = False

    def build_from_corpus(self, tokenized_sentences):
        """Build vocabulary from list of tokenized sentences."""
        for tokens in tokenized_sentences:
            self.word_freq.update(tokens)

        # Filter by frequency
        qualified = [
            (word, freq) for word, freq in self.word_freq.most_common()
            if freq >= self.min_freq
        ]

        # Apply max vocab size
        if self.max_vocab_size:
            qualified = qualified[:self.max_vocab_size - len(self.word2idx)]

        for word, _ in qualified:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self._built = True
        return self

    def encode(self, tokens):
        """Convert token list to index list with SOS/EOS."""
        indices = [self.SOS_IDX]
        indices.extend(self.word2idx.get(t, self.UNK_IDX) for t in tokens)
        indices.append(self.EOS_IDX)
        return indices

    def decode(self, indices, skip_special=True):
        """Convert index list back to token list."""
        special = {self.PAD_IDX, self.SOS_IDX, self.EOS_IDX}
        tokens = []
        for idx in indices:
            if idx == self.EOS_IDX and skip_special:
                break
            if skip_special and idx in special:
                continue
            tokens.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return tokens

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word,
                          "word_freq": self.word_freq}, f)

    @classmethod
    def load(cls, path):
        vocab = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        vocab.word_freq = data["word_freq"]
        vocab._built = True
        return vocab


# ============================================================================
# TEXT CLEANING & TOKENIZATION
# ============================================================================

def unicode_to_ascii(text):
    """Convert Unicode to ASCII, stripping accents."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize_text(text):
    """Clean and normalize raw text for NLP processing."""
    text = unicode_to_ascii(text.lower().strip())
    # Add space before punctuation
    text = re.sub(r"([.!?,;:])", r" \1", text)
    # Remove non-alphanumeric (except basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9.!?,;:'\s-]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Simple whitespace tokenizer after normalization."""
    return normalize_text(text).split()


# ============================================================================
# DATA LOADING & PAIRING
# ============================================================================

def load_document_pairs(filepath):
    """
    Load source-target text pairs from a JSON/JSONL file.
    Expected format per line: {"source": "...", "target": "..."}
    """
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            src = record.get("source", "").strip()
            tgt = record.get("target", "").strip()
            if src and tgt:
                pairs.append((src, tgt))
    return pairs


def load_plain_text_pairs(filepath, delimiter="\t"):
    """Load tab-separated source\\ttarget pairs from a text file."""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(delimiter, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def generate_synthetic_pairs(num_pairs=5000):
    """
    Generate synthetic document summarization pairs for training/demo.
    Simulates: long document -> concise summary
    """
    import random
    random.seed(42)

    templates = {
        "financial": {
            "sources": [
                "The quarterly financial report for {company} indicates revenue of ${revenue}M, "
                "representing a {growth}% {direction} year over year. Operating expenses "
                "{expense_trend} to ${expenses}M. Net income was ${income}M. The board "
                "approved a dividend of ${dividend} per share. Management projects continued "
                "{outlook} in the coming quarters driven by {driver}.",

                "Annual earnings for {company} reached ${revenue}M with {growth}% growth. "
                "The company reported operating margins of {margin}% and free cash flow of "
                "${fcf}M. Capital expenditures totaled ${capex}M as the firm invested in "
                "{investment}. The CFO noted that {metric} improved by {improvement}% "
                "compared to the prior period.",
            ],
            "targets": [
                "{company} reported ${revenue}M revenue, {direction} {growth}% YoY with "
                "${income}M net income.",

                "{company} earnings hit ${revenue}M, {growth}% growth, {margin}% operating "
                "margin.",
            ],
        },
        "product": {
            "sources": [
                "The {product} by {company} features a {spec1} {component1}, {spec2} "
                "{component2}, and {spec3} {component3}. It is designed for {audience} "
                "who need {use_case}. Available in {colors}, the device weighs {weight}g "
                "and includes {feature1} and {feature2}. Pricing starts at ${price}.",

                "Introducing the {product} from {company}. Built with {spec1} {component1} "
                "and {spec2} {component2}, this product delivers {benefit1} and {benefit2}. "
                "The {product} supports {protocol} connectivity and ships with {accessory}. "
                "Target release date is {quarter} {year} at ${price}.",
            ],
            "targets": [
                "{company} {product}: {spec1} {component1}, {spec2} {component2}, "
                "from ${price}.",

                "New {product} by {company} with {spec1} {component1}, launching {quarter} "
                "at ${price}.",
            ],
        },
        "research": {
            "sources": [
                "This study examines the relationship between {variable1} and {variable2} "
                "using a dataset of {n} observations from {source}. We employ {method} "
                "to analyze {aspect}. Results indicate a {strength} {correlation} "
                "(p < {pvalue}). The findings suggest that {implication}. Limitations "
                "include {limitation}. Future work should explore {future}.",

                "We present a {approach} for {task} that achieves {metric} of {score} on "
                "the {benchmark} benchmark. Our model uses {architecture} with "
                "{technique} to process {input_type}. Compared to prior work by {author}, "
                "our approach improves {improvement_area} by {delta}%. We release our "
                "code and pretrained weights for reproducibility.",
            ],
            "targets": [
                "Study finds {strength} {correlation} between {variable1} and {variable2} "
                "(p < {pvalue}) using {method}.",

                "{approach} for {task} achieves {score} {metric} on {benchmark}, "
                "improving {improvement_area} by {delta}%.",
            ],
        },
    }

    fillers = {
        "company": ["Acme Corp", "TechNova", "GlobalTech", "DataStream", "CloudPeak",
                     "NexGen AI", "Quantum Labs", "Vertex Inc", "Pinnacle Systems", "HorizonX"],
        "revenue": [str(round(random.uniform(50, 5000), 1)) for _ in range(20)],
        "growth": [str(round(random.uniform(-5, 35), 1)) for _ in range(20)],
        "direction": ["increase", "decrease", "improvement", "growth"],
        "expense_trend": ["increased", "decreased", "remained stable", "grew moderately"],
        "expenses": [str(round(random.uniform(20, 2000), 1)) for _ in range(20)],
        "income": [str(round(random.uniform(5, 800), 1)) for _ in range(20)],
        "dividend": [str(round(random.uniform(0.25, 5.0), 2)) for _ in range(20)],
        "outlook": ["growth", "expansion", "market penetration", "margin improvement"],
        "driver": ["cloud adoption", "AI integration", "market expansion", "cost optimization"],
        "margin": [str(round(random.uniform(8, 45), 1)) for _ in range(20)],
        "fcf": [str(round(random.uniform(10, 500), 1)) for _ in range(20)],
        "capex": [str(round(random.uniform(5, 300), 1)) for _ in range(20)],
        "investment": ["R&D", "cloud infrastructure", "AI capabilities", "global expansion"],
        "metric": ["revenue per employee", "customer retention", "ARPU", "gross margin"],
        "improvement": [str(round(random.uniform(2, 25), 1)) for _ in range(20)],
        "product": ["ProMax X1", "AirWave 5G", "SmartHub", "DataLink Pro", "NeuralCore",
                     "PixelView Ultra", "StreamBox", "QuantumChip Z", "EcoSmart", "HyperDrive"],
        "spec1": ["8-core", "12GB", "5nm", "256GB", "4K HDR", "WiFi 7", "120Hz", "64MP"],
        "component1": ["processor", "RAM", "chipset", "storage", "display", "module",
                        "panel", "sensor"],
        "spec2": ["6000mAh", "1TB", "OLED", "Titanium", "IP68", "USB-C", "DDR5", "8K"],
        "component2": ["battery", "SSD", "screen", "chassis", "rating", "port", "memory",
                        "camera"],
        "spec3": ["AI-powered", "liquid cooling", "carbon fiber", "solar-assisted"],
        "component3": ["assistant", "system", "body", "charging"],
        "audience": ["professionals", "gamers", "content creators", "enterprise users"],
        "use_case": ["high performance computing", "mobile productivity",
                      "real-time analytics", "creative workflows"],
        "colors": ["Black, Silver, and Blue", "White and Space Gray", "Midnight and Starlight"],
        "weight": [str(random.randint(150, 800)) for _ in range(10)],
        "feature1": ["fast charging", "wireless charging", "noise cancellation"],
        "feature2": ["biometric auth", "5G support", "spatial audio"],
        "price": [str(random.randint(99, 2999)) for _ in range(20)],
        "benefit1": ["exceptional performance", "all-day battery life", "seamless connectivity"],
        "benefit2": ["enterprise security", "cloud integration", "AI acceleration"],
        "protocol": ["Bluetooth 5.3", "WiFi 6E", "5G NR", "Thread/Matter"],
        "accessory": ["premium carrying case", "USB-C cable", "wireless charger"],
        "quarter": ["Q1", "Q2", "Q3", "Q4"],
        "year": ["2024", "2025"],
        "variable1": ["employee satisfaction", "remote work frequency",
                       "training investment", "team diversity"],
        "variable2": ["productivity", "innovation output", "retention rate",
                       "revenue growth"],
        "n": [str(random.randint(500, 50000)) for _ in range(10)],
        "source": ["Fortune 500 companies", "healthcare providers",
                    "educational institutions", "tech startups"],
        "method": ["regression analysis", "neural network classification",
                    "Bayesian inference", "transformer-based NLP"],
        "aspect": ["temporal patterns", "causal relationships",
                    "distributional properties", "latent factors"],
        "strength": ["strong positive", "moderate negative", "significant",
                      "weak but significant"],
        "correlation": ["correlation", "association", "relationship", "dependency"],
        "pvalue": ["0.001", "0.01", "0.05", "0.005"],
        "implication": ["targeted interventions improve outcomes",
                        "the effect is moderated by industry sector",
                        "early adoption correlates with better results"],
        "limitation": ["sample size constraints", "self-reported data",
                        "cross-sectional design", "geographic bias"],
        "future": ["longitudinal studies", "causal inference methods",
                    "multi-modal data integration", "real-time prediction"],
        "approach": ["transformer-based approach", "graph neural network",
                      "hybrid CNN-LSTM model", "attention-augmented method"],
        "task": ["document summarization", "text classification",
                  "named entity recognition", "question answering"],
        "score": [str(round(random.uniform(0.75, 0.98), 3)) for _ in range(10)],
        "benchmark": ["GLUE", "SQuAD 2.0", "CNN/DailyMail", "XSum", "MMLU"],
        "architecture": ["multi-head attention", "bidirectional encoding",
                          "hierarchical decoding", "sparse attention"],
        "technique": ["Bahdanau attention", "copy mechanism",
                       "beam search decoding", "label smoothing"],
        "input_type": ["long documents", "structured tables",
                        "multi-turn dialogues", "noisy web text"],
        "author": ["Vaswani et al.", "Devlin et al.", "Liu et al.", "Brown et al."],
        "improvement_area": ["ROUGE-L", "F1 score", "BLEU", "inference speed"],
        "delta": [str(round(random.uniform(1, 15), 1)) for _ in range(10)],
    }

    pairs = []
    categories = list(templates.keys())

    for i in range(num_pairs):
        cat = random.choice(categories)
        src_tmpl = random.choice(templates[cat]["sources"])
        tgt_tmpl = random.choice(templates[cat]["targets"])

        fill = {k: random.choice(v) for k, v in fillers.items()}
        try:
            src = src_tmpl.format(**fill)
            tgt = tgt_tmpl.format(**fill)
            pairs.append((src, tgt))
        except (KeyError, IndexError):
            continue

    return pairs


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_dataset(pairs, src_vocab=None, tgt_vocab=None,
                    max_src_len=150, max_tgt_len=60,
                    min_freq=2, max_vocab_size=10000):
    """
    Full preprocessing pipeline:
    1. Tokenize source and target texts
    2. Build vocabularies (or use provided ones)
    3. Encode sequences to integer indices
    4. Filter by length constraints
    """
    # Tokenize
    tokenized_pairs = []
    for src, tgt in pairs:
        src_tokens = tokenize(src)
        tgt_tokens = tokenize(tgt)
        if src_tokens and tgt_tokens:
            tokenized_pairs.append((src_tokens, tgt_tokens))

    print(f"  Tokenized {len(tokenized_pairs)} pairs from {len(pairs)} raw pairs")

    # Build vocabularies
    if src_vocab is None:
        src_vocab = Vocabulary(min_freq=min_freq, max_vocab_size=max_vocab_size)
        src_vocab.build_from_corpus([p[0] for p in tokenized_pairs])

    if tgt_vocab is None:
        tgt_vocab = Vocabulary(min_freq=min_freq, max_vocab_size=max_vocab_size)
        tgt_vocab.build_from_corpus([p[1] for p in tokenized_pairs])

    print(f"  Source vocab: {len(src_vocab)} tokens | "
          f"Target vocab: {len(tgt_vocab)} tokens")

    # Encode and filter
    encoded_pairs = []
    filtered_count = 0
    for src_tokens, tgt_tokens in tokenized_pairs:
        if len(src_tokens) > max_src_len or len(tgt_tokens) > max_tgt_len:
            filtered_count += 1
            continue
        src_ids = src_vocab.encode(src_tokens)
        tgt_ids = tgt_vocab.encode(tgt_tokens)
        encoded_pairs.append((src_ids, tgt_ids))

    print(f"  Encoded {len(encoded_pairs)} pairs "
          f"(filtered {filtered_count} exceeding length limits)")

    return encoded_pairs, src_vocab, tgt_vocab


def pad_sequence(seq, max_len, pad_value=0):
    """Pad a sequence to max_len."""
    return seq[:max_len] + [pad_value] * max(0, max_len - len(seq))


def create_batches(encoded_pairs, batch_size=32, pad_idx=0):
    """Create padded batches from encoded pairs."""
    import random
    # Sort by source length for efficient packing
    sorted_pairs = sorted(encoded_pairs, key=lambda x: len(x[0]))

    batches = []
    for i in range(0, len(sorted_pairs), batch_size):
        batch = sorted_pairs[i:i + batch_size]
        src_seqs = [p[0] for p in batch]
        tgt_seqs = [p[1] for p in batch]

        max_src = max(len(s) for s in src_seqs)
        max_tgt = max(len(t) for t in tgt_seqs)

        padded_src = [pad_sequence(s, max_src, pad_idx) for s in src_seqs]
        padded_tgt = [pad_sequence(t, max_tgt, pad_idx) for t in tgt_seqs]
        src_lengths = [len(s) for s in src_seqs]

        batches.append({
            "src": padded_src,
            "tgt": padded_tgt,
            "src_lengths": src_lengths,
            "batch_size": len(batch),
        })

    # Shuffle batches (not within batch, to keep similar lengths together)
    random.shuffle(batches)
    return batches


def save_processed_data(encoded_pairs, src_vocab, tgt_vocab, output_dir):
    """Save preprocessed data and vocabularies to disk."""
    os.makedirs(output_dir, exist_ok=True)

    src_vocab.save(os.path.join(output_dir, "src_vocab.pkl"))
    tgt_vocab.save(os.path.join(output_dir, "tgt_vocab.pkl"))

    with open(os.path.join(output_dir, "encoded_pairs.pkl"), "wb") as f:
        pickle.dump(encoded_pairs, f)

    # Save stats
    stats = {
        "num_pairs": len(encoded_pairs),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "avg_src_len": sum(len(p[0]) for p in encoded_pairs) / len(encoded_pairs),
        "avg_tgt_len": sum(len(p[1]) for p in encoded_pairs) / len(encoded_pairs),
        "max_src_len": max(len(p[0]) for p in encoded_pairs),
        "max_tgt_len": max(len(p[1]) for p in encoded_pairs),
    }
    with open(os.path.join(output_dir, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved processed data to {output_dir}")
    print(f"  Stats: {json.dumps(stats, indent=2)}")

    return stats
