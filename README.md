

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

<img width="474" height="534" alt="image" src="https://github.com/user-attachments/assets/cc9c0b71-d5af-439a-9c83-f271666edf8a" />


<img width="1536" height="542" alt="image" src="https://github.com/user-attachments/assets/344f134a-ba0e-4ce0-9640-e892f4c1a6d0" />

```

```


## 2- MasterEmbedding with Rotary Positional Encoding


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

3. **UstaEmbedding Layer:**

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

embedding_layer = UstaEmbedding(vocab_size=1000, embedding_dim=64, context_length=32, device="cpu")
tokens = torch.randint(0, 1000, (2, 32))  # batch of 2 sequences
output = embedding_layer(tokens)           # output shape: (2, 32, 64)
```

---

 **Summary:**
`UstaEmbedding` takes token IDs and outputs **dense, position-aware embeddings**. The combination of **semantic embeddings** (`nn.Embedding`) and **rotary positional encoding** allows transformer models to understand **both what each token means and where it appears in a sequence**.




