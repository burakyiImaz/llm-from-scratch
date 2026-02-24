import torch
import torch.nn as nn
import torch.optim as optim
import math
import os


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device="cpu",
        lr=3e-4,
        weight_decay=0.01,
        epochs=10,
        warmup_steps=500,
        grad_clip=1.0,
        save_dir="training_outputs",
        early_stopping_patience=3,
        gradient_accumulation_steps=1,
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
        self.gradient_accumulation_steps = gradient_accumulation_steps

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.total_steps = len(train_loader) * epochs
        self.global_step = 0
        self.base_lr = lr

        self.loss_fn = nn.CrossEntropyLoss()

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        # AMP Setup
        self.use_amp = device in ["cuda", "mps"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

        print(f"Toplam parametre: {sum(p.numel() for p in model.parameters()):,}")


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
        return self.loss_fn(
            logits.reshape(B * T, V),
            targets.reshape(B * T)
        )


    def compute_accuracy(self, logits, targets):
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).float()
        return correct.mean().item()


    def train_epoch(self, epoch):

        self.model.train()
        total_loss = 0
        total_acc = 0

        self.optimizer.zero_grad()

        for step, (x, y) in enumerate(self.train_loader):

            x = x.to(self.device)
            y = y.to(self.device)

            with torch.autocast(
                device_type=self.device,
                dtype=torch.float16,
                enabled=self.use_amp
            ):
                logits = self.model(x)
                loss = self.compute_loss(logits, y)
                acc = self.compute_accuracy(logits, y)

            if torch.isnan(loss):
                print("NaN detected. Training stopped.")
                return None, None

            loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                self.global_step += 1
                self.update_learning_rate()

            total_loss += loss.item() * self.gradient_accumulation_steps
            total_acc += acc

            if step % 200 == 0:
                print(
                    f"Epoch {epoch+1} | "
                    f"Step {step}/{len(self.train_loader)} | "
                    f"Loss {loss.item() * self.gradient_accumulation_steps:.4f}"
                )

        return (
            total_loss / len(self.train_loader),
            total_acc / len(self.train_loader)
        )


    def evaluate(self, loader):

        self.model.eval()
        total_loss = 0
        total_acc = 0

        with torch.no_grad():
            for x, y in loader:

                x = x.to(self.device)
                y = y.to(self.device)

                with torch.autocast(
                    device_type=self.device,
                    dtype=torch.float16,
                    enabled=self.use_amp
                ):
                    logits = self.model(x)
                    loss = self.compute_loss(logits, y)
                    acc = self.compute_accuracy(logits, y)

                total_loss += loss.item()
                total_acc += acc

        avg_loss = total_loss / len(loader)
        perplexity = math.exp(avg_loss)

        return avg_loss, perplexity, total_acc / len(loader)


    def save_checkpoint(self, name, epoch):

        path = os.path.join(self.save_dir, name)

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "early_stop_counter": self.early_stop_counter
        }, path)

        print(f"Model kaydedildi: {path}")


    def load_checkpoint(self, path):

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.early_stop_counter = checkpoint["early_stop_counter"]

        print("Checkpoint yüklendi.")



    def train(self):

        print("Training başladı\n")

        for epoch in range(self.epochs):

            train_loss, train_acc = self.train_epoch(epoch)

            if train_loss is None:
                break

            val_loss, val_ppl, val_acc = self.evaluate(self.val_loader)

            print(f"\nEpoch {epoch+1}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f}")
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

        test_loss, test_ppl, test_acc = self.evaluate(self.test_loader)

        print("\nFinal Test Sonuçları")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc: {test_acc:.4f}")
        print(f"Test Perplexity: {test_ppl:.2f}")

        self.save_checkpoint("last_model.pt", epoch)