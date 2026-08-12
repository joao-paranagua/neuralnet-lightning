import re
import numpy as np
import pandas as pd

from .cnn2d import PreprocessCNN2D

class PreprocessFused:
    # Casa apenas cl_ring_<numero>, ignorando os anéis restritos
    RING_PATTERN = re.compile(r'^cl_ring_(\d+)$')

    def __init__(self, n_rings=100, ring_norm='norm1'):
        """
        Pré-processador do modelo fusionado.
        Reaproveita o PreprocessCNN2D para as células e trata os anéis aqui.
        """
        self.cells_pp = PreprocessCNN2D()
        self.n_rings = n_rings
        self.ring_norm = ring_norm

    def find_ring_columns(self, df):
        """Localiza as colunas dos anéis e as ordena pelo índice numérico."""
        found = {}
        for col in df.columns:
            m = self.RING_PATTERN.match(str(col))
            if m:
                found[int(m.group(1))] = col

        indices = sorted(found)
        if indices != list(range(self.n_rings)):
            raise ValueError(f"Esperava cl_ring_0..cl_ring_{self.n_rings - 1}, encontrei {len(indices)} anéis.")

        return [found[i] for i in indices]

    def process_rings(self, df):
        """Pré-processa os anéis. Retorna (N, n_rings)."""
        X = df[self.find_ring_columns(df)].to_numpy(dtype=np.float32, copy=True)

        # Trata anomalias de sensores com erro no root (-999)
        X = np.where(X == -999, 0.0, X)

        if self.ring_norm == 'norm1':
            # Normalização padrão do Ringer: divide pela energia total do anelamento
            total = np.abs(X).sum(axis=1, keepdims=True)
            np.divide(X, total, out=X, where=total > 0)
        elif self.ring_norm == 'log':
            X = np.log1p(np.clip(X, 0, None))

        return X

    def transform_split(self, df):
        """Retorna anéis e células separados."""
        return self.process_rings(df), self.cells_pp.transform(df)

    def transform(self, df):
        """Concatena anéis e células achatadas em um único array (N, 835)."""
        X_rings, X_cells = self.transform_split(df)
        X_cells_flat = X_cells.reshape(X_cells.shape[0], -1)
        return np.concatenate([X_rings, X_cells_flat], axis=1).astype(np.float32)

    def get_labels(self, df, label_col='label'):
        """Retorna os labels."""
        if label_col in df.columns:
            return df[label_col].values.astype(np.float32)
        return None
