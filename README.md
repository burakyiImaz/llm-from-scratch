#LLM-FROM-SCRATCH
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

