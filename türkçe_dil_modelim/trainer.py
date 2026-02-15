import torch
import torch.optim import optim
import torch.nn as nn

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
        self.model= model.to(device)
        self.dataloader=dataloader
        self.device= device
        self.lr= lr 
        self.epochs= epochs
        self.grad_clip= grad_clip
        self.save_path= save_path

        self.optim= optim.AdamW(self.model.parameters(),lr=lr)
        self.loss_fn= nn.CrossEntropyLoss()

    def train_one_epoch(self,epoch):
        
        self.model.train()
        total_loss= 0

        for step, (x,y) in enumerate(self.dataloader):
            x= x.to(device)
            y= y.to(device)

            logits = self.model(x)  # (B, T, V)
            B, T, V = logits.shape

            loss = self.loss_fn(
                logits.view(B*T, V),
                y.view(B*T)
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.dataloader)
        return avg_loss
    
    def train(self):
        for epoch in range(self.epochs):
            avg_loss = self.train_one_epoch(epoch)
            print(f"\n Epoch {epoch} Ortalama Loss: {avg_loss:.4f}\n")

        self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model kaydedildi: {self.save_path}")
