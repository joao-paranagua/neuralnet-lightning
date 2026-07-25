import pandas as pd
import os

class LabelGenerator:
    """
    Módulo responsável por gerar labels para os dados do projeto.
    Regra definida:
    - 'Zee'  -> Label 1
    - 'JF17' -> Label 0
    """
    
    @staticmethod
    def get_label_from_path(file_path: str) -> int:
        """
        Retorna o label baseado no nome do arquivo ou diretório (path).
        """
        file_path_lower = file_path.lower()
        if 'zee' in file_path_lower:
            return 1
        elif 'jf17' in file_path_lower:
            return 0
        else:
            raise ValueError(f"Não foi possível determinar o label para o caminho: {file_path}")

    @classmethod
    def apply_label(cls, df: pd.DataFrame, file_path_col: str = 'file_path', label_col: str = 'label') -> pd.DataFrame:
        """
        Aplica a coluna de label ao DataFrame com base na coluna contendo os caminhos de arquivo.
        """
        if file_path_col not in df.columns:
            raise ValueError(f"A coluna '{file_path_col}' não foi encontrada no DataFrame.")
        df[label_col] = df[file_path_col].apply(cls.get_label_from_path)
        return df
