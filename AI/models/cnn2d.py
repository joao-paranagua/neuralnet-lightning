import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC

class ModelCNN2D(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        """
        Inicializa o modelo CNN 2D.
        Entrada esperada: (Batch, Canais=7, Altura=7, Largura=15)
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Arquitetura da Rede
        self.features = nn.Sequential(
            # Bloco Convolucional 1
            # Entrada: (Batch, 7, 7, 15)
            nn.Conv2d(in_channels=7, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Saída: (Batch, 32, 3, 7)
            
            # Bloco Convolucional 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Omitimos maxpool aqui pois a altura já é 3
        )
        
        # Flatten transforma (Batch, 64, 3, 7) em (Batch, 1344)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Saída bruta para o BCEWithLogitsLoss
        )

        # Métricas (para tarefa binária)
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        """Passo forward da rede."""
        x = self.features(x)
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
