# Generative AI & Large Language Models (LLM)

A comprehensive repository covering foundational sequence models, modern Transformer architectures, generative AI lifecycles, and large language model (LLM) paradigms. This project aggregates deep dive research notes, interview preparation guides, architectural taxonomies, and academic reference papers.

---

## 📂 Repository Structure

```text
├── GEN_AI/
│   └── RNN.md                                  # RNN, LSTM, GRU mechanics & interview Q&A
├── introduction_and_life_cycle/
│   ├── AI Model Architectures Compendium.pdf   # Architectural compendium reference paper
│   ├── attention is all you need.pdf           # Seminal Vaswani et al. (2017) paper
│   ├── intro.md                                # Topic roadmap & syllabus
│   └── research_notes.md                       # Graduate research notes & taxonomy analysis
├── assets/                                     # Diagrams and visual resources
└── README.md                                   # Repository overview & navigation guide
```

---

## 📚 Core Modules & Topics

### 1. Classical Sequence Modeling & Recurrence
Detailed in [`GEN_AI/RNN.md`](GEN_AI/RNN.md) and [`introduction_and_life_cycle/research_notes.md`](introduction_and_life_cycle/research_notes.md:11):
- **Recurrent Neural Networks (RNNs):** Hidden state updates, parameter sharing across time steps, Backpropagation Through Time (BPTT).
- **Recurrence Bottlenecks:** Vanishing and exploding gradient challenges, $\mathcal{O}(n)$ sequential dependencies, and information compression bottlenecks.
- **Gated Architectures:** Long Short-Term Memory (LSTM) cells (Forget, Input, Output gates) and Gated Recurrent Units (GRU) (Reset, Update gates).

### 2. Modern Transformer Architectures & Foundations
Detailed in [`introduction_and_life_cycle/research_notes.md`](introduction_and_life_cycle/research_notes.md:47):
- **Scaled Dot-Product Attention:** Differentiable content-addressable memory lookup $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$.
- **Multi-Head Attention (MHA):** Simultaneous representation subspace projections.
- **Positional Encodings:** Sinusoidal encodings, Rotary Position Embeddings (RoPE), and ALiBi.
- **Layer Normalization & Residuals:** Pre-LN vs. Post-LN stability in deep networks.
- **Position-wise Feedforward Networks (FFN):** SwiGLU, GELU, and ReLU activations.

### 3. Neural Architecture Taxonomies
Detailed in [`introduction_and_life_cycle/research_notes.md`](introduction_and_life_cycle/research_notes.md:93):
- **Encoder-Only:** Bidirectional representations for classification, embeddings, and extractive QA (e.g., BERT, RoBERTa).
- **Decoder-Only:** Causal autoregressive language models for next-token generation and in-context reasoning (e.g., GPT-3/4, Llama 2/3, Mistral).
- **Encoder-Decoder:** Sequence-to-sequence translation and abstractive summarization (e.g., T5, BART).
- **Vision Transformers (ViT) & Vision-Language Models (VLM):** Patch-based image modeling (ViT) and multimodal cross-attention alignment (CLIP, LLaVA).
- **Mixture of Experts (MoE):** Sparse routing for conditional computation scaling (e.g., Mixtral 8x7B).
- **Diffusion Models & Large Action Models (LAMs):** Score matching for synthesis and ReAct agent loops for action execution.

### 4. Algorithmic & Computational Complexity Analysis
Detailed in [`introduction_and_life_cycle/research_notes.md`](introduction_and_life_cycle/research_notes.md:139):
- **Self-Attention vs. Recurrence vs. Convolutions:** Computational complexity per layer, maximum path lengths, and KV cache memory scaling.

---

## 📖 Primary Reference Documents

- **Attention Is All You Need** (Vaswani et al., 2017) — [`introduction_and_life_cycle/attention is all you need.pdf`](introduction_and_life_cycle/attention%20is%20all%20you%20need.pdf)
- **AI Model Architectures Compendium** — [`introduction_and_life_cycle/AI Model Architectures Compendium.pdf`](introduction_and_life_cycle/AI%20Model%20Architectures%20Compendium.pdf)
- **Curriculum & Roadmap Overview** — [`introduction_and_life_cycle/intro.md`](introduction_and_life_cycle/intro.md)

---

## 🎯 Interview & Study Guides

- **RNN & Sequence Modeling Interview Guide:** Fast-review summaries, mathematical formulas, comparative tables, and high-frequency interview Q&A in [`GEN_AI/RNN.md`](GEN_AI/RNN.md:130).
- **Mathematical Formulations & Architectural Proofs:** In-depth derivations and theoretical breakdowns in [`introduction_and_life_cycle/research_notes.md`](introduction_and_life_cycle/research_notes.md).
