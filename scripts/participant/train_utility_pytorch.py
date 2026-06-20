import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
from pathlib import Path
from agent.utility_model_pytorch import PhaseUtilityNet

class PairwiseUtilityDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        print(f"Parsing {len(df)} rows from CSV...")
        
        self.state_feats = []
        self.action_feats = []
        self.teacher_scores = []
        self.safe_masks = []
        self.teacher_margins = []
        self.actual_wins = []
        self.action_takens = []
        
        # Determine the win outcome column
        win_col = "win" if "win" in df.columns else "actual_win"
        
        for idx, row in df.iterrows():
            try:
                state_feat = json.loads(row["state_features"])
                action_feat = json.loads(row["action_features"])
                teacher_score = json.loads(row["teacher_scores"])
                safe_mask = json.loads(row["safe_mask"])
                action_taken = int(row["action_taken"])
                
                self.state_feats.append(state_feat)
                self.action_feats.append(action_feat)
                self.teacher_scores.append(teacher_score)
                self.safe_masks.append(safe_mask)
                self.teacher_margins.append(float(row["teacher_margin"]))
                self.actual_wins.append(float(row[win_col]))
                self.action_takens.append(action_taken)
            except Exception as e:
                print(f"Error parsing row {idx}: {e}")
                continue
                
        self.state_feats = torch.tensor(self.state_feats, dtype=torch.float32)
        self.action_feats = torch.tensor(self.action_feats, dtype=torch.float32)
        self.teacher_scores = torch.tensor(self.teacher_scores, dtype=torch.float32)
        self.safe_masks = torch.tensor(self.safe_masks, dtype=torch.float32)
        self.teacher_margins = torch.tensor(self.teacher_margins, dtype=torch.float32)
        self.actual_wins = torch.tensor(self.actual_wins, dtype=torch.float32)
        self.action_takens = torch.tensor(self.action_takens, dtype=torch.long)
        
        print(f"Successfully loaded {len(self.state_feats)} samples.")

    def __len__(self):
        return len(self.state_feats)

    def __getitem__(self, idx):
        return (
            self.state_feats[idx],
            self.action_feats[idx],
            self.teacher_scores[idx],
            self.safe_masks[idx],
            self.teacher_margins[idx],
            self.actual_wins[idx],
            self.action_takens[idx]
        )

def train_hybrid_ranker(dataset_path, output_pth_path, epochs=15, batch_size=256, lr=1e-3):
    print(f"\n--- Training PyTorch Expert Imitation Model ---")
    print(f"Dataset path: {dataset_path}")
    print(f"Output model path: {output_pth_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = PairwiseUtilityDataset(dataset_path)
    if len(dataset) < 10:
        print("Error: Too few samples to train.")
        return
        
    # Split train/validation (85/15)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = PhaseUtilityNet(state_dim=13, action_dim=33).to(device)
    
    # We use Cross Entropy Loss for imitation learning (Behavior Cloning)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for state, action, teacher, safe, margin, win, action_taken in train_loader:
            state = state.to(device)
            action = action.to(device)
            action_taken = action_taken.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass outputs the absolute utility for all actions
            pred = model(state, action)  # [batch, 6]
            
            # Cross Entropy Loss matches model's predictions with expert choices
            loss = criterion(pred, action_taken)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * len(state)
            
        scheduler.step()
        train_loss = epoch_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for state, action, teacher, safe, margin, win, action_taken in val_loader:
                state = state.to(device)
                action = action.to(device)
                action_taken = action_taken.to(device)
                
                pred = model(state, action)
                loss = criterion(pred, action_taken)
                val_loss_sum += loss.item() * len(state)
                
                # Check accuracy
                preds = pred.argmax(dim=-1)
                correct += (preds == action_taken).sum().item()
                total += len(state)
                
        val_loss = val_loss_sum / len(val_dataset)
        val_acc = float(correct) / total * 100.0
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_pth_path)
            print(f"  --> Saved new best model to {output_pth_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train imitation model.")
    parser.add_argument("--dataset_path", type=str, default="agent/expert_utility_dataset.csv", help="Path to CSV dataset.")
    parser.add_argument("--output_path", type=str, default="agent/learned_utility_model.pth", help="Path to output model.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()
    
    train_hybrid_ranker(
        dataset_path=args.dataset_path,
        output_pth_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

if __name__ == "__main__":
    main()
