import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CLS(nn.Module):
    def __init__(self, input_dim, lr=1e-3, weight_decay=1e-4, n_epochs=1000, device="cpu"):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # model: Wx + b
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.device = device
        self.to(device)
        self.training_loss = []

    def fit(self, X, y):
        """
        X: numpy or torch tensor, shape [N, D]
        y: numpy or torch tensor, shape [N] or [N, 1], values in {0,1}
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)

        X = X.to(self.device)
        y = y.view(-1, 1).to(self.device)

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in tqdm.tqdm(range(self.n_epochs)):
            optimizer.zero_grad()
            logits = self.linear(X)                     # raw scores
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            self.training_loss.append(loss.item())
            
        self.loss_ = loss.item()
        return self

    def predict_proba(self, X):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.linear(X)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs > 0.5).astype("float32")

    def score(self, X, y):
        preds = self.predict(X).reshape(-1)
        return (preds == y).mean()

    @property
    def coef_(self):
        return self.linear.weight.detach().cpu().numpy()

    @property
    def intercept_(self):
        return self.linear.bias.detach().cpu().numpy()
