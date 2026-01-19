import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class NumberSequenceDataset(Dataset):
    def __init__(self, sequences, number_to_id):
        self.sequences = sequences
        self.number_to_id = number_to_id
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_seq = torch.tensor([self.number_to_id[n] for n in seq[:-1]], dtype=torch.long)
        target_seq = torch.tensor([self.number_to_id[n] for n in seq[1:]], dtype=torch.long)
        return input_seq, target_seq

class NumberPredictorModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)  
        out, _ = self.lstm(x)  
        logits = self.fc(out)   
        return logits

class NumberPredictor:
    def __init__(self, embed_dim=64, hidden_dim=128, lr=0.001, batch_size=32, epochs=10):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.model = None
        self.number_to_id = {}
        self.id_to_number = {}
        self.vocab_size = 0

    def prepare_vocab(self, sequences):
        all_numbers = set()
        for seq in sequences:
            all_numbers.update(seq)
        self.number_to_id = {num: idx for idx, num in enumerate(sorted(all_numbers))}
        self.id_to_number = {idx: num for num, idx in self.number_to_id.items()}
        self.vocab_size = len(self.number_to_id)

    def train(self, sequences):
        self.prepare_vocab(sequences)
        dataset = NumberSequenceDataset(sequences, self.number_to_id)
        
        def collate_fn(batch):
            inputs, targets = zip(*batch)
            return (
                torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0),
                torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
            )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

        
        self.model = NumberPredictorModel(self.vocab_size, self.embed_dim, self.hidden_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs.view(-1, self.vocab_size), targets.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {total_loss/len(dataloader):.4f}")

    def predict_next(self, input_seq, top_k=5):
        if not self.model:
            raise ValueError("[!] The model is not trained or loaded.")
        self.model.eval()
        with torch.no_grad():
            ids = [self.number_to_id[n] for n in input_seq]
            input_tensor = torch.tensor([ids], dtype=torch.long)
            outputs = self.model(input_tensor)
            last_logits = outputs[0, -1]
            probs = F.softmax(last_logits, dim=0).cpu().numpy()

        top_indices = probs.argsort()[-top_k:][::-1]
        return {self.id_to_number[i]: float(probs[i]) for i in top_indices}

    def save(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'number_to_id': self.number_to_id,
            'id_to_number': self.id_to_number,
            'vocab_size': self.vocab_size,
            'config': {
                'embed_dim': self.embed_dim,
                'hidden_dim': self.hidden_dim,
                'lr': self.lr,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }, path)
        print(f"[*]  Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.number_to_id = checkpoint['number_to_id']
        self.id_to_number = checkpoint['id_to_number']
        self.vocab_size = checkpoint['vocab_size']
        cfg = checkpoint['config']
        self.embed_dim = cfg['embed_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.lr = cfg['lr']
        self.batch_size = cfg['batch_size']
        self.epochs = cfg['epochs']
        self.model = NumberPredictorModel(self.vocab_size, self.embed_dim, self.hidden_dim)
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"[*] Model loaded from {path}")

    def sequence_log_probability(self, seq):
        if not self.model:
            raise ValueError("[!] The model is not trained or loaded.")
        self.model.eval()
        log_prob_sum = 0.0

        with torch.no_grad():
            for i in range(1, len(seq)):
                input_seq = seq[:i]
                next_num = seq[i]
                ids = [self.number_to_id[n] for n in input_seq]
                input_tensor = torch.tensor([ids], dtype=torch.long)
                outputs = self.model(input_tensor)
                last_logits = outputs[0, -1]
                log_probs = F.log_softmax(last_logits, dim=0).cpu().numpy()
                next_num_id = self.number_to_id[next_num]
                log_prob_next_num = log_probs[next_num_id]
                log_prob_sum += log_prob_next_num

        return log_prob_sum
    
    def generate_sequences(self, start_seq, length=5, top_k=5):
        sequences = [(start_seq, 0.0)]

        for _ in range(len(start_seq), length):
            all_candidates = []
            for seq, seq_log_prob in sequences:
                next_preds = self.predict_next(seq, top_k=top_k)  # {num: prob}
                for num, prob in next_preds.items():
                    new_seq = seq + [num]
                    # avoid overfitting
                    log_prob_step = torch.log(torch.tensor(prob))
                    all_candidates.append((new_seq, seq_log_prob + log_prob_step.item()))

            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = all_candidates[:top_k]

        return sequences
    
    def count_series(self, series):
        serie = pd.Series(series)
        count = serie.value_counts(normalize=True, sort=True)
        return count