import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import time

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device="cpu",
        lr=2e-4,
        weight_decay=0.01,
        epochs=10,
        warmup_steps=2000,
        grad_clip=0.5,
        save_dir="training_outputs",
        early_stopping_patience=3
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.early_stopping_patience = early_stopping_patience

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # Parametreleri weight decay ve no decay olarak ayırıyoruz
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        self.optimizer = optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr
        )

        self.total_steps = len(train_loader) * epochs
        self.global_step = 0
        self.base_lr = lr

        # Label smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        # Mixed precision scaler (MPS/CUDA için)
        self.scaler = torch.cuda.amp.GradScaler(enabled=device in ["cuda","mps"])


    def update_learning_rate(self):
        if self.global_step < self.warmup_steps:
            lr = self.base_lr * self.global_step / max(1, self.warmup_steps)
        else:
            progress = (self.global_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


    def compute_loss(self, logits, targets):
        B, T, V = logits.shape
        return self.loss_fn(logits.view(-1, V), targets.view(-1))



    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        epoch_start = time.time()

        for step, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device,
                dtype=torch.float16,
                enabled=self.device in ["cuda", "mps"]
            ):
                logits = self.model(x)
                loss = self.compute_loss(logits, y)

            # Mixed precision
            if self.device in ["cuda", "mps"]:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.global_step += 1
            self.update_learning_rate()

            total_loss += loss.item()

            # ---- Progress Logging ----
            if step % 100 == 0 and step > 0:
                elapsed = time.time() - epoch_start

                steps_done = epoch * len(self.train_loader) + step
                total_steps = self.total_steps

                progress = steps_done / total_steps * 100

                tokens_per_batch = x.numel()
                tokens_processed = steps_done * tokens_per_batch
                tok_per_sec = tokens_processed / max(elapsed, 1)

                remaining_steps = total_steps - steps_done
                eta_sec = remaining_steps * (elapsed / max(steps_done, 1))
                eta_min = eta_sec / 60

                print(
                    f"Epoch {epoch+1} | Step {step}/{len(self.train_loader)} | "
                    f"Loss {loss.item():.4f} | "
                    f"Progress {progress:.2f}% | "
                    f"{tok_per_sec:,.0f} tok/s | "
                    f"ETA {eta_min:.1f} min"
                )

        return total_loss / len(self.train_loader)


    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.autocast(
                    device_type=self.device,
                    dtype=torch.float16,
                    enabled=self.device in ["cuda", "mps"]
                ):
                    logits = self.model(x)
                    loss = self.compute_loss(logits, y)

                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        perplexity = float(torch.exp(torch.tensor(avg_loss)))
        return avg_loss, perplexity


    def save_checkpoint(self, name, epoch):
        path = os.path.join(self.save_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step
            },
            path
        )
        print(f"Model kaydedildi: {path}")


    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        print("Checkpoint yüklendi.")


    def train(self):
        print("Training başladı\n")

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_ppl = self.evaluate(self.val_loader)

            print(f"\nEpoch {epoch+1}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Perplexity: {val_ppl:.2f}\n")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt", epoch)
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.early_stopping_patience:
                print("Early stopping tetiklendi.")
                break

        test_loss, test_ppl = self.evaluate(self.test_loader)
        print("\nFinal Test Sonuçları")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Perplexity: {test_ppl:.2f}")

        self.save_checkpoint("last_model.pt", epoch)