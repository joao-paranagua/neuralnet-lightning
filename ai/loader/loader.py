import pandas as pd
import glob
import os
from tqdm import tqdm

class DataLoader:
    def __init__(self, data_path=None, max_files=None):
        self.max_files = max_files
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(script_dir, "..", "..", "data", "parquet", "**", "*.parquet")
        else:
            self.data_path = data_path

    def get_files(self):
        # Se o caminho fornecido for uma pasta (comum para datasets particionados), busca os .parquet dentro
        if os.path.isdir(self.data_path):
            search_path = os.path.join(self.data_path, "**", "*.parquet")
        else:
            search_path = self.data_path
            
        arquivos_por_pasta = {}
        for f in glob.glob(search_path, recursive=True):
            if os.path.isfile(f):
                pasta = os.path.dirname(f)
                if pasta not in arquivos_por_pasta:
                    arquivos_por_pasta[pasta] = []
                arquivos_por_pasta[pasta].append(f)
        
        arquivos = []
        for pasta, arqs in arquivos_por_pasta.items():
            arqs.sort()
            if self.max_files is not None:
                arquivos.extend(arqs[:self.max_files])
            else:
                arquivos.extend(arqs)
                
        print(f"Encontrados {len(arquivos)} arquivos válidos")
        return arquivos

    def load_dataset(self, arquivos):
        if not arquivos:
            print("Nenhum arquivo encontrado para carregar.")
            return None
        
        dfs = []
        for f in tqdm(arquivos, desc="Carregando Parquets", unit="arquivo"):
            df_temp = pd.read_parquet(f)
            df_temp['file_path'] = f
            dfs.append(df_temp)
            
        df = pd.concat(dfs, ignore_index=True)
        return df

    def execute(self):
        arquivos = self.get_files()
        df = self.load_dataset(arquivos)
        return df

if __name__ == "__main__":
    loader = DataLoader()
    loader.execute()