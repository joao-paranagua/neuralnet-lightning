import numpy as np

class DataBalancer:
    def __init__(self, random_state=42):
        """
        Inicializa o balanceador de dados.
        Por padrão, utiliza a técnica de Undersampling (reduz a classe majoritária
        para o tamanho da classe minoritária).
        """
        self.random_state = random_state

    def apply(self, X, Y):
        """
        Aplica o balanceamento sobre os dados X e as labels Y.
        Pressupõe que Y é um array 1D com os rótulos de classe (ex: 0 e 1).
        
        Args:
            X (np.ndarray): Array de features.
            Y (np.ndarray): Array de labels.
            
        Returns:
            tuple: (X_balanced, Y_balanced)
        """
        # Garante que Y seja tratado como um array plano para a contagem,
        # mas mantendo o formato original de Y no retorno, se necessário.
        y_flat = Y.flatten() if Y.ndim > 1 else Y
        
        classes, counts = np.unique(y_flat, return_counts=True)
        
        if len(classes) < 2:
            print("Aviso (DataBalancer): Menos de 2 classes encontradas. O balanceamento foi ignorado.")
            return X, Y

        min_count = np.min(counts)
        print(f"DataBalancer: Balanceando para {min_count} amostras por classe (Undersampling)...")
        
        rng = np.random.default_rng(seed=self.random_state)
        
        balanced_indices = []
        for cls in classes:
            # Pega os índices onde a classe atual ocorre
            cls_indices = np.where(y_flat == cls)[0]
            # Seleciona aleatoriamente 'min_count' índices dessa classe
            selected_indices = rng.choice(cls_indices, size=min_count, replace=False)
            balanced_indices.append(selected_indices)
            
        # Concatena todos os índices selecionados
        balanced_indices = np.concatenate(balanced_indices)
        
        # Embaralha os índices finais para que as classes não fiquem agrupadas (ex: 000111)
        rng.shuffle(balanced_indices)
        
        # Retorna o subconjunto de dados balanceado
        return X[balanced_indices], Y[balanced_indices]
