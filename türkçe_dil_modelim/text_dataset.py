import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TextDataset(Dataset):
    def __init__(self, token_ids, context_length, stride):
        super().__init__()
        self.inputs = []
        self.targets = []

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i : i + context_length]
            target_chunk = token_ids[i + 1 : i + context_length + 1]

            if len(input_chunk) < context_length or len(target_chunk) < context_length:
                continue

            self.inputs.append(torch.tensor(input_chunk, dtype=torch.long))
            self.targets.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    # ---------------------------
    # Static Utility Functions
    # ---------------------------

    @staticmethod
    def load_encoded_file(file_path):
        all_tokens = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_tokens = [int(tok) for tok in line.strip().split()]
                all_tokens.extend(line_tokens)

        return all_tokens

    @staticmethod
    def process_and_save_batches(
        file_path,
        tokenizer,
        context_length=32,
        stride=1,
        batch_size=64,
        save_dir="wikipedia_batches",
        chunk_size=5000
    ):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        batch_files = []
        chunk_tokens = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                tokens = tokenizer.encode(line)
                chunk_tokens.extend(tokens.tolist())

                if (line_idx + 1) % chunk_size == 0:
                    print(f"{line_idx+1} satır işlendi")
                    batch_files.extend(
                        TextDataset.save_chunk_batches(
                            chunk_tokens,
                            context_length,
                            stride,
                            batch_size,
                            save_dir,
                            line_idx
                        )
                    )
                    chunk_tokens = []

        if len(chunk_tokens) > context_length:
            batch_files.extend(
                TextDataset.save_chunk_batches(
                    chunk_tokens,
                    context_length,
                    stride,
                    batch_size,
                    save_dir,
                    line_idx
                )
            )

        tokenizer.save_vocab()
        print(f"Tüm batchler kaydedildi: {len(batch_files)} dosya")
        return batch_files

    @staticmethod
    def save_chunk_batches(
        token_ids_chunk,
        context_length,
        stride,
        batch_size,
        save_dir,
        line_idx
    ):
        dataset = TextDataset(token_ids_chunk, context_length, stride)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        files = []

        for batch_idx, (x, y) in enumerate(loader):
            batch_file = save_dir / f"batch_{line_idx}_{batch_idx}.pt"
            torch.save({"input": x, "target": y}, batch_file)
            files.append(batch_file)

        return files
