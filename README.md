

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


Got it 👍 Here’s an English version you can directly put into your README.

---

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


<img width="350" height="600" alt="image" src="https://github.com/user-attachments/assets/9521bcb9-d9d7-40ae-bfb2-8ef4de7a298f" img width="350" height="600" alt="image" src="https://github.com/user-attachments/assets/3d352aef-0dbf-4292-8957-41290755e12d" >








