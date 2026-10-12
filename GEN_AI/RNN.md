# Recurrent Neural Networks (RNN) — Interview Notes

## 1. What is an RNN?

A **Recurrent Neural Network (RNN)** is a type of neural network designed to process **sequential data**, where the order of information matters.

Unlike traditional feedforward neural networks, an RNN maintains a **hidden state (memory)** that carries information from previous time steps to help process the current input.

### Key Points
- Designed for sequential data such as text, speech, and time-series data.
- Each RNN cell processes the current input and previous hidden state.
- Passes the updated hidden state to the next time step.
- Uses shared weights across all time steps.
- Can capture dependencies between sequential inputs.

---

## 2. How Does an RNN Work?

An RNN processes input sequentially, one step at a time.

Example: "I love AI"

1. Step 1: Processes "I" and generates a hidden state.
2. Step 2: Processes "love" using the previous hidden state.
3. Step 3: Processes "AI" using the updated hidden state.

Each RNN cell combines:
- Current input (`x_t`)
- Previous hidden state (`h_{t-1}`)

To generate:
- Updated hidden state (`h_t`)

The hidden state acts as the network's memory.

---

## 3. RNN Mathematical Formula

The hidden state is calculated as:

`h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)`

### Parameters

| Parameter | Description |
|-----------|-------------|
| `x_t` | Current input |
| `h_{t-1}` | Previous hidden state |
| `W_x` | Input weight matrix |
| `W_h` | Recurrent weight matrix |
| `b` | Bias |
| `tanh` | Activation function |
| `h_t` | Updated hidden state |

**Key Interview Point:**

The previous hidden state (`h_{t-1}`) allows the RNN to retain information from earlier time steps.

---

## 4. Limitations of Basic RNNs

Basic RNNs struggle to learn **long-term dependencies**.

### Vanishing Gradient Problem

During training, RNNs use **Backpropagation Through Time (BPTT)**.

As gradients propagate backward through multiple time steps, they may become extremely small.

This causes:
- Difficulty learning long-term dependencies.
- Weak learning signals for earlier inputs.
- Reduced ability to capture relationships across long sequences.

### Other Limitations
- Exploding gradients.
- Sequential processing limits parallelization.
- Difficulty retaining information across long sequences.

---

## 5. Improvements Over Basic RNNs

### LSTM — Long Short-Term Memory

- Advanced RNN architecture.
- Designed to learn long-term dependencies.
- Uses a cell state and gating mechanisms.
- Three main gates:
  - Forget gate
  - Input gate
  - Output gate

### GRU — Gated Recurrent Unit

- Simplified version of gated RNN architecture.
- Uses fewer gates than LSTM.
- Two main gates:
  - Update gate
  - Reset gate
- Generally has fewer parameters than an equivalent LSTM.

### Transformers

- Use attention mechanisms instead of recurrence.
- Can model relationships between distant tokens.
- Enable parallel processing of input tokens during training.
- Form the foundation of modern GPT-style LLMs.

---

## 6. RNN vs Transformer

| Feature | RNN | Transformer |
|---------|-----|-------------|
| Architecture | Recurrent | Attention-based |
| Processing | Sequential | Parallelizable during training |
| Memory | Hidden state | Attention over context tokens |
| Long-range dependencies | Difficult for basic RNNs | Generally better |
| Training parallelization | Limited | Highly parallelizable |
| Modern LLM usage | Uncommon | Dominant architecture |

**Note:** Autoregressive Transformers still generate output tokens sequentially during inference.

---

## 7. Interview-Ready Answer (30–45 Seconds)

A Recurrent Neural Network (RNN) is a neural network designed to process sequential data.

Unlike traditional feedforward networks, an RNN maintains a hidden state that carries information from previous time steps.

At each step, the RNN combines the current input with the previous hidden state to generate an updated hidden state, allowing it to model sequential dependencies.

However, basic RNNs struggle with long-term dependencies due to the vanishing gradient problem during training.

To address this limitation, architectures such as LSTM and GRU introduced gating mechanisms.

Modern Transformers use self-attention instead of recurrence, enabling better training parallelization and effective modeling of long-range relationships.

---

## 8. Common Interview Questions

### Q1. What is a hidden state in an RNN?
A hidden state is a vector that carries information from previous time steps and is updated when new input is processed.

### Q2. What is the vanishing gradient problem?
During backpropagation, gradients can become extremely small, making it difficult for the model to learn long-term dependencies.

### Q3. What is Backpropagation Through Time (BPTT)?
BPTT is the training procedure that unfolds an RNN across time steps and propagates gradients backward through those steps.

### Q4. What is the difference between LSTM and GRU?
LSTM uses three main gates and a separate cell state, while GRU uses two main gates and does not maintain a separate cell state.

### Q5. Why are Transformers preferred over RNNs in modern LLMs?
Transformers use self-attention, which enables efficient parallel training and generally handles long-range token relationships more effectively.

---

## 9. Quick Revision

**RNN = Sequential Processing + Hidden State + Shared Weights**

**Basic RNN Limitation = Vanishing Gradients + Long-Term Dependency Challenges**

**Improvements = LSTM + GRU**

**Modern LLM Architecture = Transformer + Self-Attention**