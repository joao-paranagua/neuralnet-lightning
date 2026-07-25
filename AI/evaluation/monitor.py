import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import mplhep as hep
plt.style.use(hep.style.ATLAS)

class ModelMonitor:
    def __init__(self, output_dir="results/plots"):
        """
        Inicializa o monitor para plotar métricas de avaliação do modelo.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_roc_curve(self, y_true, y_prob, filename="roc_curve.pdf"):
        """Plota e salva a Curva ROC com a métrica AUC."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (FPR)')
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Curva ROC salva em: {os.path.join(self.output_dir, filename)}")
        
    def plot_confusion_matrix(self, y_true, y_pred, filename="confusion_matrix.pdf"):
        """Plota e salva a Matriz de Confusão do modelo."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel('Classe Prevista (Predicted)')
        plt.ylabel('Classe Real (True)')
        plt.title('Matriz de Confusão')
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Matriz de Confusão salva em: {os.path.join(self.output_dir, filename)}")

    def plot_loss(self, train_loss, val_loss, filename="loss_curve.pdf"):
        """Plota a curva de aprendizado contendo a loss de treino e validação ao longo das épocas."""
        plt.figure(figsize=(8, 6))
        plt.plot(train_loss, label='Treino (Train Loss)', linewidth=2)
        plt.plot(val_loss, label='Validação (Val Loss)', linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('Função de Custo (Loss)')
        plt.title('Curva de Aprendizado')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Curva de Custo (Loss) salva em: {os.path.join(self.output_dir, filename)}")
