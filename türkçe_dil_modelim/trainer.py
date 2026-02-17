import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        device="cpu",
        lr=3e-4,
        epochs=5,
        grad_clip=1.0,
        save_path="turkce_llm.pt"
    ):
        self.device = device
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.save_path = save_path

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_one_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(self.dataloader):

            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)  # (B, T, V)
            B, T, V = logits.shape

            loss = self.loss_fn(
                logits.view(B * T, V),
                y.view(B * T)
            )

            self.optimizer.zero_grad()
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
                    f"Step {step}/{len(self.dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(self.dataloader)
        return avg_loss

    def save_checkpoint(self, epoch, step):
        path = f"checkpoints/model_epoch{epoch}_step{step}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": step
        }, path)
        print(f"Model kaydedildi: {path}")

    def train(self):

        print(" Training başladı...\n")

        for epoch in range(self.epochs):
            avg_loss = self.train_one_epoch(epoch)
            print(f"\n Epoch {epoch+1} Ortalama Loss: {avg_loss:.4f}\n")

        self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)
        print(f" Model kaydedildi: {self.save_path}")
