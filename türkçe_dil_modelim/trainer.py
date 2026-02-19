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
        epochs=20,
        warmup_steps=500,
        grad_clip=1.0,
        save_dir="training_outputs"
    ):

        self.device = device
        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.epochs = epochs
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        total_steps = len(train_loader) * epochs

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.best_val_loss = float("inf")


    def compute_loss(self, logits, targets):
        B, T, V = logits.shape
        return self.loss_fn(
            logits.view(B * T, V),
            targets.view(B * T)
        )


    def train_epoch(self, epoch):

        self.model.train()
        total_loss = 0

        for step, (x, y) in enumerate(self.train_loader):

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.compute_loss(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if step % 200 == 0:
                print(
                    f"Epoch {epoch+1} | "
                    f"Step {step}/{len(self.train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        return total_loss / len(self.train_loader)


    def evaluate(self, loader):

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.compute_loss(logits, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        perplexity = math.exp(avg_loss)

        return avg_loss, perplexity


    def save_checkpoint(self, name):
        path = os.path.join(self.save_dir, name)
        torch.save(self.model.state_dict(), path)
        print(f"Model kaydedildi: {path}")


    def train(self):

        print("Training başladı\n")

        for epoch in range(self.epochs):

            train_loss = self.train_epoch(epoch)
            val_loss, val_ppl = self.evaluate(self.val_loader)

            print(f"\nEpoch {epoch+1}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Perplexity: {val_ppl:.2f}\n")

            # Best model kaydet
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")

        # Final test evaluation
        test_loss, test_ppl = self.evaluate(self.test_loader)

        print("\nFinal Test Sonuçları")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Perplexity: {test_ppl:.2f}")

        self.save_checkpoint("last_model.pt")
