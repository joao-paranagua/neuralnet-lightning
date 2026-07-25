import os
import sys
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Adiciona o diretório raiz ao path para garantir que as importações funcionem se o script for rodado de qualquer lugar
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from AI.loader.loader import DataLoader
from AI.label.label_generator import LabelGenerator
from AI.preprocess.cnn2d import PreprocessCNN2D
from AI.models.cnn2d import ModelCNN2D
from AI.trainer.trainer import ModelTrainer
from AI.evaluation.monitor import ModelMonitor
from AI.evaluation.summary import ModelSummary

class PipelineCNN2D:
    def __init__(self, data_path=None, max_files=None, label_col='label', model_name="CNN2D", max_epochs=20, batch_size=32, patience=5, num_workers=0):
        """
        Inicializa o pipeline completo para o modelo CNN 2D.
        """
        self.model_name = model_name
        self.label_col = label_col
        
        # Configuração de diretórios centralizada por nome de modelo
        self.results_dir = os.path.join("results", self.model_name)
        
        self.loader = DataLoader(data_path=data_path, max_files=max_files)
        self.preprocessor = PreprocessCNN2D()
        
        # O trainer vai salvar os checkpoints dentro de results/CNN2D/lightning_logs
        self.trainer = ModelTrainer(
            max_epochs=max_epochs, 
            batch_size=batch_size,
            patience=patience,
            num_workers=num_workers,
            log_dir=os.path.join(self.results_dir, "lightning_logs")
        )
        
        # Avaliadores configurados para pastas específicas do modelo
        self.monitor = ModelMonitor(output_dir=os.path.join(self.results_dir, "plots"))
        self.summary = ModelSummary(output_dir=os.path.join(self.results_dir, "metrics"))

    def evaluate_model(self, model, X_test, Y_test, threshold=0.5, suffix=""):
        """Avalia o modelo treinado em dados invisíveis e gera relatórios."""
        suffix_print = f" ({suffix})" if suffix else ""
        print(f"\n-> Etapa 4: Avaliando o modelo {self.model_name} (Threshold={threshold}){suffix_print}...")
        
        model.eval() # Modo inferência, desativa Dropout/BatchNorm
        with torch.no_grad():
            X_tensor = torch.as_tensor(X_test, dtype=torch.float32)
            # Passa no modelo e aplica sigmoide para pegar probabilidades (0 a 1)
            logits = model(X_tensor)
            y_prob = torch.sigmoid(logits).numpy().flatten()
            
        y_true = Y_test.flatten()
        y_pred = (y_prob >= threshold).astype(int)
        
        file_suffix = f"_{suffix}" if suffix else ""
        
        print(f"Salvando CSV de métricas em {self.summary.output_dir}...")
        self.summary.save_metrics(y_true, y_prob, threshold=threshold, filename=f"test_metrics{file_suffix}.csv")
        
        print(f"Salvando Gráficos (ROC, Confusion Matrix) em {self.monitor.output_dir}...")
        self.monitor.plot_roc_curve(y_true, y_prob, filename=f"roc_curve{file_suffix}.pdf")
        self.monitor.plot_confusion_matrix(y_true, y_pred, filename=f"confusion_matrix{file_suffix}.pdf")
        print(f"Avaliação completa{suffix_print}!")

    def run(self, use_kfold=False, n_splits=5, test_size=0.15, learning_rate=0.001, threshold=0.5, target_fold=None):
        """
        Executa todas as etapas do pipeline de ponta a ponta.
        """
        print(f"Iniciando Pipeline Completa: {self.model_name}")
        
        # Etapa 1: Carregamento
        print("\n-> Etapa 1: Carregando dados...")
        df = self.loader.execute()
        
        if df is None or df.empty:
            print("Erro: Nenhum dado foi carregado.")
            return None
            
        # Etapa 1.5: Geração de Labels
        print("\n-> Geração de Labels...")
        df = LabelGenerator.apply_label(df, file_path_col='file_path', label_col=self.label_col)
        df.drop(columns=['file_path'], inplace=True)
            
        # Etapa 2: Pré-processamento
        print("\n-> Etapa 2: Pré-processamento...")
        X = self.preprocessor.transform(df)
        Y = self.preprocessor.get_labels(df, label_col=self.label_col)
        
        if Y is None:
            print("Erro: Labels não encontrados.")
            return None
            
        # Separa um Test Set absoluto (holdout verdadeiro) para a Etapa 4 de Avaliação
        # Este dado NUNCA é visto pelo Trainer.
        print(f"Separando {test_size*100}% dos dados para Teste Isolado...")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        
        # Etapa 3: Treinamento
        print(f"\n-> Etapa 3: Treinamento (K-Fold={use_kfold})...")
        
        if use_kfold:
            model_kwargs = {'learning_rate': learning_rate}
            fold_trainers, fold_models = self.trainer.fit_kfold(
                ModelCNN2D, model_kwargs, X_train, Y_train, 
                n_splits=n_splits, target_fold=target_fold
            )
            print("\n[Aviso K-Fold]: A avaliação gráfica será gerada individualmente para cada fold treinado!")
            for i, model in enumerate(fold_models):
                # Se for rodado um fold específico, a lista só tem 1 elemento, então o sufixo é o target_fold
                fold_idx = target_fold if target_fold is not None else (i + 1)
                self.evaluate_model(model, X_test, Y_test, threshold=threshold, suffix=f"fold_{fold_idx}")
            return fold_trainers, fold_models
        else:
            # Treina normal
            model = ModelCNN2D(learning_rate=learning_rate)
            trained_trainer = self.trainer.fit(model, X_train, Y_train)
            
            # Etapa 4: Avaliação
            # Analisa o modelo usando o test_set puro isolado.
            self.evaluate_model(model, X_test, Y_test, threshold=threshold)
            
            return model, trained_trainer
