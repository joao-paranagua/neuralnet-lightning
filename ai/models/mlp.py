import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC

class ModelMLP(pl.LightningModule):
    def __init__(self, input_dim=100, learning_rate=0.001):
        """
        Inicializa o modelo MLP.
        Entrada esperada: (Batch, input_dim)
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Arquitetura da Rede
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),  # Camada de entrada (100) -> Oculta (128)
            nn.ReLU(),
            nn.Dropout(0.5),            # Dropout para evitar overfitting
            nn.Linear(128, 64),         # Oculta (128) -> Oculta (64)
            nn.ReLU(),
            nn.Linear(64, 1)            # Saída (64) -> Logit binário (1)
        )

        # Métricas
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        """Passo forward da rede."""
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()
        
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits)
        
        self.train_acc(preds, y)
        self.train_auc(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()
        
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits)
        
        self.val_acc(preds, y)
        self.val_auc(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_auc', self.val_auc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
