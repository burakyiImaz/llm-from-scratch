Here’s a clean **Day 1 README in English** for your tokenizer project. I kept it short and beginner-friendly since you’ll add more details day by day.

````markdown
# MasterTokenizer — Day 1 README (EN)

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

Below diagram shows the workflow of the tokenizer using `tokenizer.json`:

![Tokenizer Flow](images/tokenizer_flow.png)

```

---

Do you want me to also generate the **`tokenizer_flow.png`** diagram for you, so you can just drop it into an `images/` folder in your repo?
```
