import numpy as np

class PreprocessMLP:
    def __init__(self, num_rings=100):
        """
        Inicializa o pré-processador para a MLP.
        Espera receber as colunas de anéis do Lorenzetti.
        """
        self.num_rings = num_rings
        self.ring_columns = [f"cl_truth_ring_{i}" for i in range(self.num_rings)]

    def transform(self, df):
        """Transforma o DataFrame em tensores PyTorch"""

        missing = [col for col in self.ring_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Faltam colunas de anéis: {missing}")

        print("\nExtraindo anéis (Rings)...")
        X = df[self.ring_columns].values.astype(np.float32)
        
        # Trata anomalias de sensores com erro no root (-999)
        X = np.where(X == -999, 0, X)

        # Clip e transformação logarítmica para comprimir caudas de energia
        X = np.log1p(np.clip(X, 0, None))
        
        return X

    def get_labels(self, df, label_col='has_truth_clus'):
        """Retorna os labels."""
        if label_col in df.columns:
            return df[label_col].values.astype(np.float32)
        return None
