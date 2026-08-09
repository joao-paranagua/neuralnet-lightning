import argparse
import yaml
import sys
import os

# Garante que o Python encontre o pacote ai a partir da raiz do projeto
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Orquestrador de Treinamento de Redes Neurais (ATLAS CERN).")
    parser.add_argument('--config', type=str, default='config.yaml', help="Caminho para o arquivo YAML de configuração.")
    parser.add_argument('--fold', type=int, default=None, help="Executar apenas um fold específico (útil para paralelismo via SLURM).")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Erro: Arquivo de configuração '{args.config}' não encontrado.")
        sys.exit(1)

    print(f"Carregando configurações de: {args.config}")
    config = load_config(args.config)
    
    model_type = config.get("model", "CNN2D")
    
    if model_type == "CNN2D":
        from ai.pipeline.pipeline_cnn2d import PipelineCNN2D
        pipeline = PipelineCNN2D(
            data_path=config.get("data_path"),
            max_files=config.get("max_files"),
            label_col=config.get("label_col", "label"),
            model_name=model_type,
            max_epochs=config.get("max_epochs", 20),
            batch_size=config.get("batch_size", 32),
            patience=config.get("patience", 5),
            num_workers=config.get("num_workers", 0)
        )
    # Adicionar o elif model_type == "CNN1D" aqui no futuro
    else:
        raise ValueError(f"Modelo '{model_type}' não é suportado ou ainda não foi implementado na pipeline.")

    # Inicia o fluxo completo de carregamento, treinamento e validação
    pipeline.run(
        use_kfold=config.get("use_kfold", False),
        n_splits=config.get("n_splits", 5),
        test_size=config.get("test_size", 0.15),
        learning_rate=config.get("learning_rate", 0.001),
        threshold=config.get("threshold", 0.5),
        target_fold=args.fold
    )

if __name__ == "__main__":
    main()
