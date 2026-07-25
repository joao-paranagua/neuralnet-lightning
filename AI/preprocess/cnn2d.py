import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas(desc="Processando Amostras")

class PreprocessCNN2D:
    def __init__(self, target_shape=(7, 7, 15)):
        """
        Inicializa o pré-processador para a CNN 2D.
        target_shape: (Canais, Altura_Max, Largura_Max) -> (7 camadas, max_h=7, max_w=15)
        """
        self.target_shape = target_shape
        self.cell_columns = [
            'cl_cells_presampler', # Original: (3, 3)
            'cl_cells_em1',        # Original: (3, 15)
            'cl_cells_em2',        # Original: (7, 7)
            'cl_cells_em3',        # Original: (5, 5)
            'cl_cells_had1',       # Original: (5, 5)
            'cl_cells_had2',       # Original: (5, 5)
            'cl_cells_had3'        # Original: (5, 5)
        ]
        self.max_h = 7
        self.max_w = 15

    def pad_array(self, arr):
        """Pré-processa uma camada (imagem 2D) de energia."""
        # O Parquet retorna um array de arrays (dtype=object). Empilhamos para formar uma matriz 2D Float32.
        arr = np.stack(arr).astype(np.float32)
        
        # Trata anomalias de sensores com erro no root (-999)
        arr = np.where(arr == -999, 0, arr)
        
        # Clip e transformação logarítmica para comprimir caudas de energia
        arr = np.log1p(np.clip(arr, 0, None))
        
        h, w = arr.shape
        pad_h = self.max_h - h
        pad_w = self.max_w - w
        
        # Calcula os paddings para centralizar a matriz menor dentro da matriz (7, 15)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)

    def transform(self, df):
        """Transforma o DataFrame em tensores PyTorch."""
        missing = [col for col in self.cell_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Faltam colunas: {missing}")

        num_samples = len(df)
        num_layers = len(self.cell_columns)
        
        # Formato PyTorch: (Batch, Channels, Height, Width) -> (N, 7, 7, 15)
        X = np.zeros((num_samples, num_layers, self.max_h, self.max_w), dtype=np.float32)

        print("\nConvertendo camadas do calorímetro em Imagens 2D (Tensores)...")
        for i, col in enumerate(self.cell_columns):
            print(f"[{i+1}/{num_layers}] Processando canal: {col}")
            # Processa e empilha todas as imagens 2D dessa camada para todos os eventos
            layer_arrays = np.stack(df[col].progress_apply(self.pad_array).values)
            
            # Insere no canal 'i' (representando a profundidade do calorímetro)
            X[:, i, :, :] = layer_arrays
            
        return X

    def get_labels(self, df, label_col='has_truth_clus'):
        """Retorna os labels."""
        if label_col in df.columns:
            return df[label_col].values.astype(np.float32)
        return None
