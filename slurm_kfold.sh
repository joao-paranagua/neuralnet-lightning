#!/bin/bash

# ==============================================================================
# Orquestrador Paralelo de Validação Cruzada (K-Fold) via SLURM
# ==============================================================================

# Máquinas GPU disponíveis na sua fila/reserva atual
MACHINES=("caloba68" "caloba69" "caloba70" "caloba71")

# Parâmetros de entrada com valores padrão
NUM_MACHINES=${1:-4}        # Padrão: usar até 4 máquinas
NUM_FOLDS=${2:-3}           # Padrão: rodar 3 folds (deve bater com o yaml)
CONFIG_FILE=${3:-"config.yaml"}

# Evita tentar usar mais máquinas do que as disponíveis no array
if [ "$NUM_MACHINES" -gt "${#MACHINES[@]}" ]; then
    echo "[Aviso]: Solicitado $NUM_MACHINES máquinas, mas só há ${#MACHINES[@]} definidas. Ajustando..."
    NUM_MACHINES=${#MACHINES[@]}
fi

echo "====================================================================="
echo "Iniciando submissão massivamente paralela via srun (SLURM)"
echo "Folds a calcular: $NUM_FOLDS"
echo "Máquinas em uso: $NUM_MACHINES"
echo "Arquivo de Configuração: $CONFIG_FILE"
echo "====================================================================="

# Cria pasta para os logs do SLURM (arquivos .out e .err)
LOG_DIR="results/logs_slurm"
mkdir -p "$LOG_DIR"

# Loop disparando 1 srun em background para CADA fold
for (( fold=1; fold<=NUM_FOLDS; fold++ )); do
    
    # Lógica Round-Robin: se tiver 3 máquinas e 5 folds, ele envia:
    # Fold 1 -> Mach 0, Fold 2 -> Mach 1, Fold 3 -> Mach 2
    # Fold 4 -> Mach 0, Fold 5 -> Mach 1
    MACHINE_IDX=$(( (fold - 1) % NUM_MACHINES ))
    MACHINE=${MACHINES[$MACHINE_IDX]}
    
    echo "Submetendo Fold $fold na máquina: $MACHINE..."
    
    # Dispara o job via sbatch (fila do SLURM) usando a configuração da sua reserva
    sbatch -p gpu --reservation=gdi -N 1 -w "$MACHINE" \
         --job-name="CNN_2D_Cern_fold_${fold}" \
         --output="${LOG_DIR}/fold_${fold}_%j.out" \
         --error="${LOG_DIR}/fold_${fold}_%j.err" \
         --wrap="python ai/run.py --config $CONFIG_FILE --fold $fold"
         
    # Pequeno delay apenas para evitar concorrência no gerenciador de filas
    sleep 1
done

echo "====================================================================="
echo "Todos os $NUM_FOLDS folds foram submetidos para processamento paralelo via fila!"
echo "Use 'squeue -u \$USER' para monitorar o andamento."
echo "Para cancelar tudo, rode 'scancel -u \$USER'."
echo "====================================================================="
echo "Confira os gráficos em 'results/' e os logs em '${LOG_DIR}/' quando acabarem."
