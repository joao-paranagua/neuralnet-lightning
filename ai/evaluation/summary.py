import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

class ModelSummary:
    def __init__(self, output_dir="results/metrics"):
        """
        Inicializa o sumário para calcular e salvar as principais métricas de avaliação.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_metrics(self, y_true, y_prob, threshold=0.5, filename="metrics_summary.csv"):
        """
        Calcula as métricas Accuracy, AUC, Precision, Recall e F1-Score 
        e anexa os resultados em um arquivo CSV.
        """
        # Converte a probabilidade na classe binária usando o threshold (padrão = 0.5)
        y_pred = (y_prob >= threshold).astype(int)
        
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1_Score": f1_score(y_true, y_pred, zero_division=0)
        }
        
        df = pd.DataFrame([metrics])
        filepath = os.path.join(self.output_dir, filename)
        
        # Se o arquivo já existe, anexa os novos resultados. Se não, cria com cabeçalho
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
            print(f"Métricas anexadas ao arquivo: {filepath}")
        else:
            df.to_csv(filepath, index=False)
            print(f"Métricas salvas no novo arquivo: {filepath}")
            
        return metrics
