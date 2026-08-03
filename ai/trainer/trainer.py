import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
from sklearn.model_selection import KFold

class ModelTrainer:
    def __init__(self, max_epochs=20, batch_size=32, validation_split=0.2, patience=5, log_dir="lightning_logs", num_workers=0):
        """
        Inicializa o Trainer Genérico para modelos PyTorch Lightning.
        
        Args:
            max_epochs: Número máximo de épocas.
            batch_size: Tamanho do batch.
            validation_split: Fração dos dados usada para validação (se não usar k-fold).
            patience: Quantidade de épocas sem melhoria antes de parar (EarlyStopping).
            log_dir: Diretório onde salvar os pesos e logs do modelo.
            num_workers: Quantidade de processos paralelos para carregar dados (0 = thread principal).
        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.log_dir = log_dir
        self.num_workers = num_workers

    def prepare_data(self, X, Y):
        """
        Converte arrays numpy para tensores e cria os DataLoaders de Treino e Validação simples.
        """
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X, dtype=torch.float32)
        if isinstance(Y, np.ndarray):
            Y = torch.as_tensor(Y, dtype=torch.float32)

        dataset = TensorDataset(X, Y)
        
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        return train_loader, val_loader

    def fit(self, model, X, Y):
        """
        Treina qualquer modelo (LightningModule) usando holdout simples (treino/validação única).
        """
        print("Preparando DataLoaders para Holdout...")
        train_loader, val_loader = self.prepare_data(X, Y)
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=self.patience, mode="min", verbose=True),
            ModelCheckpoint(
                dirpath=self.log_dir, 
                monitor="val_loss", 
                save_top_k=1, 
                mode="min",
                filename=f"{model.__class__.__name__}-{{epoch:02d}}-{{val_loss:.4f}}"
            )
        ]
        
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            accelerator="auto",
            devices="auto",
            default_root_dir=self.log_dir
        )
        
        print(f"Iniciando treinamento do modelo {model.__class__.__name__}...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        print(f"Treinamento finalizado! Melhor modelo: {trainer.checkpoint_callback.best_model_path}")
        return trainer

    def fit_kfold(self, model_class, model_kwargs, X, Y, n_splits=5, target_fold=None):
        """
        Executa validação cruzada (K-Fold).
        Cria uma nova instância do modelo a cada fold para não vazar pesos.
        
        Args:
            model_class: A classe do modelo (ex: ModelCNN2D).
            model_kwargs: Dicionário com os argumentos (ex: {'learning_rate': 1e-3}).
            X, Y: Dados de entrada.
            n_splits: Número de subdivisões.
            target_fold: Executar apenas um fold isolado (1-indexed). Útil para SLURM.
            
        Returns:
            tuple: (fold_trainers, fold_models) Listas contendo os trainers e modelos de cada fold.
        """
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X, dtype=torch.float32)
        if isinstance(Y, np.ndarray):
            Y = torch.as_tensor(Y, dtype=torch.float32)

        dataset = TensorDataset(X, Y)
        
        # shuffle=True garante que as amostras sejam misturadas antes de fatiar
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_trainers = []
        fold_models = []
        
        print(f"Iniciando Validação Cruzada com {n_splits} folds...")
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            if target_fold is not None and (fold + 1) != target_fold:
                continue
                
            print(f"\n{'='*20} Fold {fold + 1}/{n_splits} {'='*20}")
            
            # Subsets para este fold
            train_sub = Subset(dataset, train_ids)
            val_sub = Subset(dataset, val_ids)
            
            # DataLoaders independentes para o fold
            train_loader = DataLoader(
                train_sub, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers
            )
            val_loader = DataLoader(
                val_sub, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers
            )
            
            # CRÍTICO: Instancia um modelo "zerado" para este fold
            model = model_class(**model_kwargs)
            
            # Diretório de logs específico para o fold
            fold_log_dir = os.path.join(self.log_dir, f"fold_{fold+1}")
            os.makedirs(fold_log_dir, exist_ok=True)
            
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=self.patience, mode="min", verbose=True),
                ModelCheckpoint(
                    dirpath=fold_log_dir, 
                    monitor="val_loss", 
                    save_top_k=1, 
                    mode="min",
                    filename=f"{model.__class__.__name__}-fold{fold+1}-{{epoch:02d}}-{{val_loss:.4f}}"
                )
            ]
            
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                accelerator="auto",
                devices="auto",
                default_root_dir=fold_log_dir
            )
            
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            fold_trainers.append(trainer)
            fold_models.append(model)
            
            print(f"Melhor modelo do Fold {fold + 1} salvo em: {trainer.checkpoint_callback.best_model_path}")
            
        print(f"\nValidação Cruzada de {n_splits} Folds Concluída!")
        return fold_trainers, fold_models
