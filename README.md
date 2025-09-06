

# 1-MasterTokenizer 

## Purpose
`MasterTokenizer` uses a vocabulary file (`tokenizer.json`) to split text into subwords, convert them into token IDs, apply padding to a fixed length, and decode them back to text.

## How It Works
- **Vocabulary**: Defined as `{"token": id}` in `tokenizer.json`.
- **Special tokens**:  
  - `<unk>` = unknown subword  
  - `<pad>` = padding token  
  - `" "` = space token  
- **Algorithm**:  
  1. Split the input into words by whitespace.  
  2. For each word, find the **longest matching subword** in the vocabulary (if none, use `<unk>`).  
  3. Add a space token after each word.  
  4. If the text does not end with a space, remove the last space token.

## Example
Vocabulary (excerpt):
```json
{
  "Mer": 10,
  "haba": 11,
  "dünya": 12,
  "!": 13,
  " ": 14
}
````

Input text:

```
"Merhaba dünya!"
```

Token IDs:

```
[10, 11, 14, 12, 13]
```

Decoded back:

```
"Merhaba dünya!"
```

## Visual Explanation

Below diagram shows the workflow of the tokenizer

<img width="1536" height="534" alt="image" src="https://github.com/user-attachments/assets/cc9c0b71-d5af-439a-9c83-f271666edf8a" />


<img width="1536" height="542" alt="image" src="https://github.com/user-attachments/assets/344f134a-ba0e-4ce0-9640-e892f4c1a6d0" />

```

```


# 2- MasterEmbedding with Rotary Positional Encoding


##  Overview

The module does two main things:

1. **Embedding Tokens with `nn.Embedding`:**

   * Maps token IDs (integers representing words/subwords) into **dense, trainable vectors**.
   * These vectors capture **semantic meaning**: similar tokens have vectors that are closer in space.
   * The vectors are **learned during training** via backpropagation.

2. **Adding Positional Information:**

   * Uses **rotary positional encoding** to encode the **position of each token** in the sequence.
   * This allows the model to distinguish **token order**, which is essential for language understanding.

---

##  How It Works

1. **`nn.Embedding` Step:**

   * Takes a token ID and returns the corresponding **embedding vector** from a lookup table (matrix of shape `(vocab_size, embedding_dim)`).
   * Each vector is **trainable**, so the model learns the semantic representation of tokens.

2. **Rotary Positional Encoding:**

   * Splits each embedding vector into **even and odd halves**.
   * Rotates these halves using **sine and cosine functions** based on the token's position.
   * Produces **position-aware embeddings** that encode both token meaning and order.

3. **MasterEmbedding Layer:**

   * Combines the embedding and positional encoding steps.
   * Input: token IDs
   * Output: `(batch_size, sequence_length, embedding_dim)` tensor of **position-aware embeddings**, ready for transformers.

---

##  Why It Matters

* `nn.Embedding` converts token IDs into **continuous vectors**, allowing the model to learn **relationships between tokens**.
* Rotary positional encoding introduces **sequence order information**, so the model can process sequences effectively.
* Together, this forms a **core building block for transformer-based language models**.

---

##  Example Usage

```python
import torch

embedding_layer = MasterEmbedding(vocab_size=1000, embedding_dim=64, context_length=32, device="cpu")
tokens = torch.randint(0, 1000, (2, 32))  # batch of 2 sequences
output = embedding_layer(tokens)           # output shape: (2, 32, 64)
```

---

 **Summary:**
`MasterEmbedding` takes token IDs and outputs **dense, position-aware embeddings**. The combination of **semantic embeddings** (`nn.Embedding`) and **rotary positional encoding** allows transformer models to understand **both what each token means and where it appears in a sequence**.


<img width="768" height="432" alt="image" src="https://github.com/user-attachments/assets/f48228f6-acd2-4788-b0f2-265499836fd0" />


<img width="768" height="649" alt="image" src="https://github.com/user-attachments/assets/2929a8e4-2a90-4018-96bd-4eed287aeac8" />



# 3-MasterLayerNorm 

**Layer Normalization (LN)** standardizes the activations **along the feature (embedding) dimension of each individual example/token**. The goal is to rescale values so that each vector has **zero mean** and **unit variance**, followed by a **learnable scale parameter (γ)** (and optionally a shift parameter β). This keeps activations well-behaved, leading to **more stable training** that is **independent of batch size**. Since LN does not rely on batch statistics, it behaves the same during both training and inference.

**How to think about it**
Given an embedding vector $x \in \mathbb{R}^{d}$, LN computes its mean and variance, normalizes it as

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}}
$$

and then applies scaling (and optionally shifting):

$$
y = \gamma \odot \hat{x} (+ \, \beta)
$$

Here $\varepsilon$ ensures numerical stability, and $\gamma$ (and optionally $\beta$) preserve the model’s expressiveness.

**Where it is useful**

* Transformers use LN around sublayers (Attention, Feed-Forward).
* **Pre-LN** (before the sublayer) improves gradient flow in deep networks and is widely adopted in practice.
* **Post-LN** is mathematically similar but can cause optimization issues in very deep models.

**UstaLayerNorm note**
This implementation includes the **scale (γ)** parameter. The **shift (β)** parameter can easily be added if needed.

**Key benefits**

* Works **independently of batch size**, suitable for RNNs and Transformers,
* Consistent behavior during training and inference,
* Improves optimization stability and gradient flow in deep stacks.

---

## Visual for README

**LayerNorm axis diagram:**


<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/9521bcb9-d9d7-40ae-bfb2-8ef4de7a298f" >

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/3d352aef-0dbf-4292-8957-41290755e12d" >



# 4-MasterMLP with Custom GELU

This project implements a simplified version of an MLP (Multi-Layer Perceptron) block inspired by modern Transformer architectures.
The main goal is to process the input through multiple pathways, fuse the results, and transform them back into the original space, resulting in richer and more controlled representations.

## Core Idea

Input Transformation
The input is projected through two different linear pathways:

Gate pathway: learns how much information should pass through.

Up pathway: transforms the input into a higher-dimensional hidden space.

Activation (GELU)
The gate pathway applies GELU (Gaussian Error Linear Unit), a smooth activation function.

Unlike ReLU, GELU provides a softer, probabilistic gating of information.

This results in more stable learning and better performance in deep architectures.

Fusion (Element-wise Multiplication)
The outputs of the gate and up pathways are multiplied element-wise.

This acts like a filtering mechanism, allowing the model to keep only the most useful information.

Output Transformation
The fused representation is projected back into the original embedding dimension.

Ensures that input and output shapes match.

Makes the block easy to integrate into larger architectures.

## Why This Design?

Gating mechanism → Controls the flow of information.

GELU activation → Provides smoother and more effective transformations, widely used in models like GPT and BERT.

Flexible structure → Can be applied in Transformers, vision models, or as a standalone feature extractor.

## Suggested Visuals

To make the README more clear, you can add diagrams like:

MLP Block Structure
(Input → Gate + Up projections → GELU → Multiplication/Fusion → Output projection)


GELU Activation Curve


Transformer Block Context
(Showing how MLP fits within a Transformer block)


## Summary:
This block processes data through multiple pathways, applies smooth gating with GELU, and fuses information to produce more meaningful representations. This design enhances learning capacity and enables controlled information flow in deep neural networks.



<img width="800" height="372" alt="image" src="https://github.com/user-attachments/assets/32d54717-4526-4ab1-b23d-1afdc056d118" />

<img width="800" height="675" alt="image" src="https://github.com/user-attachments/assets/34254382-1732-4681-a29d-f4a58624e71b" />

<img width="800" height="160" alt="image" src="https://github.com/user-attachments/assets/17447ef3-b9df-4578-b0b2-4e84b9988dfd" />




# 5-MasterMultiHeadAttention

##  Core Idea

Multi-Head Attention allows a model to **focus on different parts of a sequence simultaneously**. Instead of computing a single attention distribution, it uses multiple "heads," each learning a unique way to relate tokens.

Think of it like this:

* One head might focus on **short-term dependencies** (e.g., the next word).
* Another head might capture **long-range dependencies** (e.g., subject–verb agreement).
* When combined, the model has a **richer understanding** of the sequence.

---

##  How It Works

1. **Input Embeddings**
   Each token is represented as a vector of size `embedding_dim`.

2. **Linear Projections (Q, K, V)**
   The embeddings are projected into **Query (Q)**, **Key (K)**, and **Value (V)** spaces.

   * Queries ask: *"Which other tokens should I pay attention to?"*
   * Keys provide: *"Here’s my identity."*
   * Values carry: *"Here’s my content."*

3. **Scaled Dot-Product Attention**
   Attention scores are computed as:

   $$
   \text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   $$

   This tells us how much each token should pay attention to others.

4. **Multiple Heads**
   Instead of one set of Q, K, V → multiple are created (`num_heads`).
   Each head learns **different types of relationships** in parallel.

5. **Concatenation + Projection**
   All heads are concatenated and passed through a linear layer (like your `self.projection`).

6. **Causal Mask (your code)**
   In autoregressive models (like GPT), the mask ensures that **future tokens are hidden**—a token cannot "cheat" by looking ahead.

---


##  Why It Matters

* Captures **different relationships** in text sequences.
* Enables Transformers to model **contextual meaning** effectively.
* Essential for models like **BERT, GPT, and Vision Transformers**.



##  Example Visuals

<img width="900" height="1030" alt="image" src="https://github.com/user-attachments/assets/1b7be8b3-8c19-4b24-bfd4-4323092250f9" />

<img width="900" height="420" alt="image" src="https://github.com/user-attachments/assets/1374afca-b6c6-47bc-a5c1-5bba94e4b454" />


<img width="900" height="1095" alt="image" src="https://github.com/user-attachments/assets/d01b0e81-69a2-4091-9246-ef9ec1af6754" />






## 🧩 Decoder Block Diagram

```mermaid
flowchart TD
    A[Input Embeddings] --> B[Multi-Head Self-Attention]
    B --> C[Add & Norm (Residual + LayerNorm)]
    C --> D[Feed Forward Network (MLP)]
    D --> E[Add & Norm (Residual + LayerNorm)]
    E --> F[Output Representation]





