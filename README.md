

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

Below diagram shows the workflow of the tokenizer

<img width="474" height="534" alt="image" src="https://github.com/user-attachments/assets/cc9c0b71-d5af-439a-9c83-f271666edf8a" />


<img width="1536" height="542" alt="image" src="https://github.com/user-attachments/assets/344f134a-ba0e-4ce0-9640-e892f4c1a6d0" />

```

