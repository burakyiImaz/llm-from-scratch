import torch
import torch.nn as nn
import torch.optim as optim
import os
import math


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        device="cpu",
        lr=3e-4,
        epochs=5,
        grad_clip=1.0,
        save_path="turkce_llm.pt",
        checkpoint_dir="checkpoints",
        early_stopping_patience=3
    ):

        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.save_path = save_path
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0


    def compute_loss(self, logits, targets):
        B, T, V = logits.shape
        return self.loss_fn(
            logits.view(B * T, V),
            targets.view(B * T)
        )


    def train_one_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(self.train_loader):

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model(x)
                    loss = self.compute_loss(logits, y)

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x)
                loss = self.compute_loss(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )
                self.optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch+1} | "
                    f"Step {step}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss



    def evaluate(self):

        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.compute_loss(logits, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        perplexity = math.exp(avg_loss)

        return avg_loss, perplexity



    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)
        print(f"\nFinal model kaydedildi: {self.save_path}")



    def train(self):

        print("Training başladı...\n")

        for epoch in range(self.epochs):

            train_loss = self.train_one_epoch(epoch)
            self.scheduler.step()

            print(f"\nEpoch {epoch+1} Train Loss: {train_loss:.4f}")

            if self.val_loader:
                result = self.evaluate()
                if result is not None:
                    val_loss, perplexity = result

                    print(
                        f"Validation Loss: {val_loss:.4f} | "
                        f"Perplexity: {perplexity:.2f}\n"
                    )

                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                    self.save_model()
                else:
                    self.early_stop_counter += 1

                    if self.early_stop_counter >= self.early_stopping_patience:
                        print("Early stopping tetiklendi.")
                        break
            else:
                self.save_model()
