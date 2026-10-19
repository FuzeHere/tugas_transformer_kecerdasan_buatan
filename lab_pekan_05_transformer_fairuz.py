"""
Praktikum Pekan 5 — Transformer & Self-Attention
Nama: Fairuz Istighfar | NIM: 60900124044
Adaptasi dari skrip praktikum dosen: kalimat heatmap diubah ke Bahasa Indonesia.
Bobot embedding dan proyeksi diinisialisasi acak (seed 42); ini demonstrasi matematis,
bukan model bahasa terlatih.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T
    scaled_scores = scores / np.sqrt(d_k)
    if mask is not None:
        scaled_scores = np.where(mask == 1, -1e9, scaled_scores)
    attention_weights = softmax(scaled_scores, axis=-1)
    output = attention_weights @ V
    return output, attention_weights

def create_causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len)), k=1)

def run_indonesian_sentence():
    # Kalimat buatan sendiri
    tokens = ["Saya", "belajar", "Transformer", "untuk", "memahami", "konteks", "kalimat"]
    seq_len = len(tokens)
    d_model = 8
    # Embedding sintetis acak bukan embedding dari model bahasa terlatih.
    embeddings = np.random.randn(seq_len, d_model) * 0.1
    # Bobot proyeksi acak seperti demonstrasi pada skrip praktikum.
    W_q = np.random.randn(d_model, d_model) * 0.2
    W_k = np.random.randn(d_model, d_model) * 0.2
    W_v = np.random.randn(d_model, d_model) * 0.2
    Q, K, V = embeddings @ W_q, embeddings @ W_k, embeddings @ W_v
    output, attn = scaled_dot_product_attention(Q, K, V)
    print("Kalimat:", " ".join(tokens))
    print("Bentuk Q, K, V:", Q.shape, K.shape, V.shape)
    print("Bobot attention untuk setiap query:")
    for i, token in enumerate(tokens):
        j = int(np.argmax(attn[i]))
        print(f"{token:12s} paling besar memperhatikan {tokens[j]:12s}: {attn[i,j]:.4f}")
    return tokens, attn

def plot_attention_heatmap(tokens, attn, path="heatmap_transformer_fairuz.png"):
    fig, ax = plt.subplots(figsize=(9, 7), dpi=220)
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Bobot attention (probabilitas Softmax)")
    ax.set_xticks(range(len(tokens))); ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=40, ha="left")
    ax.set_yticklabels(tokens)
    ax.xaxis.tick_top()
    ax.set_xlabel("Key (kata yang diperhatikan)")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Query (kata yang sedang memperhatikan)")
    ax.set_title("Heatmap Self-Attention — Kalimat Bahasa Indonesia", pad=30)
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{attn[i,j]:.2f}", ha="center", va="center",
                    color="white" if attn[i,j] < attn.max()*0.65 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("Heatmap disimpan:", path)

def demo_causal_mask():
    tokens = ["Saya", "sedang", "belajar", "Transformer", "AI"]
    mask = create_causal_mask(len(tokens))
    scores = np.random.randn(len(tokens), len(tokens))
    masked_scores = np.where(mask == 1, -1e9, scores)
    weights = softmax(masked_scores, axis=-1)
    print("\\nCausal mask (1 = posisi masa depan disembunyikan):\\n", mask.astype(int))
    print("Bobot attention setelah causal mask:")
    print(np.round(weights, 4))
    assert np.allclose(np.triu(weights, k=1), 0)
    print("Pemeriksaan: semua bobot di atas diagonal = 0.")

if __name__ == "__main__":
    tokens, attn = run_indonesian_sentence()
    plot_attention_heatmap(tokens, attn)
    demo_causal_mask()
