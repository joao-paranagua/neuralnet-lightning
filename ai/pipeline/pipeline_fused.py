import os
import sys
import torch
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ai.loader.loader import DataLoader
from ai.label.label_generator import LabelGenerator
from ai.preprocess.fused import PreprocessFused
from ai.preprocess.balancer import DataBalancer
from ai.models.fused import ModelFused
from ai.trainer.trainer import ModelTrainer
from ai.evaluation.monitor import ModelMonitor
from ai.evaluation.summary import ModelSummary

class PipelineFused:
    def __init__(self, data_path=None, max_files=None, label_col='label', model_name="Fused", max_epochs=20, batch_size=32, patience=5, num_workers=0, balance_data=True, n_rings=100, ring_norm='norm1', fusion_source='embedding', aux_loss_weight=0.3):
        """
        Inicializa o pipeline completo para o modelo Fusionado.
        """
        self.model_name = model_name
        self.label_col = label_col
        self.n_rings = n_rings
        self.fusion_source = fusion_source
        self.aux_loss_weight = aux_loss_weight

        self.results_dir = os.path.join("results", self.model_name)

        self.loader = DataLoader(data_path=data_path, max_files=max_files)
        self.preprocessor = PreprocessFused(n_rings=n_rings, ring_norm=ring_norm)
        self.balancer = DataBalancer() if balance_data else None

        self.trainer = ModelTrainer(
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            num_workers=num_workers,
            log_dir=os.path.join(self.results_dir, "lightning_logs")
        )

        self.monitor = ModelMonitor(output_dir=os.path.join(self.results_dir, "plots"))
        self.summary = ModelSummary(output_dir=os.path.join(self.results_dir, "metrics"))

    def evaluate_model(self, model, X_test, Y_test, threshold=0.5, suffix=""):
        """Avalia o modelo treinado em dados invisíveis e gera relatórios."""
        print(f"\n-> Etapa 4: Avaliando o modelo {self.model_name}...")

        model.eval()
        with torch.no_grad():
            X_tensor = torch.as_tensor(X_test, dtype=torch.float32)
            logits = model(X_tensor)
            y_prob = torch.sigmoid(logits).numpy().flatten()

        y_true = Y_test.flatten()
        y_pred = (y_prob >= threshold).astype(int)

        file_suffix = f"_{suffix}" if suffix else ""

        self.summary.save_metrics(y_true, y_prob, threshold=threshold, filename=f"test_metrics{file_suffix}.csv")
        self.monitor.plot_roc_curve(y_true, y_prob, filename=f"roc_curve{file_suffix}.pdf")
        self.monitor.plot_confusion_matrix(y_true, y_pred, filename=f"confusion_matrix{file_suffix}.pdf")

    def run(self, use_kfold=False, n_splits=5, test_size=0.15, learning_rate=0.001, threshold=0.5, target_fold=None):
        """
        Executa todas as etapas do pipeline de ponta a ponta.
        """
        # Etapa 1: Carregamento
        print("\n-> Etapa 1: Carregando dados...")
        df = self.loader.execute()

        if df is None or df.empty:
            print("Erro: Nenhum dado foi carregado.")
            return None

        # Etapa 1.5: Geração de Labels
        df = LabelGenerator.apply_label(df, file_path_col='file_path', label_col=self.label_col)
        df.drop(columns=['file_path'], inplace=True)

        # Etapa 2: Pré-processamento (anéis e células concatenados em um único X)
        print("\n-> Etapa 2: Pré-processamento...")
        X = self.preprocessor.transform(df)
        Y = self.preprocessor.get_labels(df, label_col=self.label_col)

        if Y is None:
            print("Erro: Labels não encontrados.")
            return None

        # Etapa 2.5: Balanceamento
        if self.balancer:
            X, Y = self.balancer.apply(X, Y)

        # Separa um Test Set absoluto que NUNCA é visto pelo Trainer
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42, shuffle=True, stratify=Y)

        # Etapa 3: Treinamento
        print(f"\n-> Etapa 3: Treinamento (K-Fold={use_kfold})...")

        model_kwargs = {
            'learning_rate': learning_rate,
            'n_rings': self.n_rings,
            'fusion_source': self.fusion_source,
            'aux_loss_weight': self.aux_loss_weight
        }

        if use_kfold:
            fold_trainers, fold_models = self.trainer.fit_kfold(
                ModelFused, model_kwargs, X_train, Y_train,
                n_splits=n_splits, target_fold=target_fold
            )
            # A avaliação é gerada individualmente para cada fold treinado
            for i, model in enumerate(fold_models):
                fold_idx = target_fold if target_fold is not None else (i + 1)
                self.evaluate_model(model, X_test, Y_test, threshold=threshold, suffix=f"fold_{fold_idx}")
            return fold_trainers, fold_models
        else:
            model = ModelFused(**model_kwargs)
            trained_trainer = self.trainer.fit(model, X_train, Y_train)

            # Etapa 4: Avaliação com o test set isolado
            self.evaluate_model(model, X_test, Y_test, threshold=threshold)

            return model, trained_trainer
