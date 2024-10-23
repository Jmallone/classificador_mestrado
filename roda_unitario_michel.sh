#!/bin/bash

# Ativa o ambiente virtual
source .venv/bin/activate

# Função para exibir ajuda
exibir_ajuda() {
    echo "Uso: $0 <patch> <feature> [<start_point>]"
    echo ""
    echo "Descrição: Este script executa um fluxo de trabalho específico com base nos parâmetros fornecidos."
    echo ""
    echo "Parâmetros:"
    echo "  <patch>        - O valor do patch."
    echo "  <feature>      - O modelo de recurso."
    echo "  <start_point>  - (opcional) O ponto de partida no fluxo de trabalho."
    echo "                   Opções disponíveis: verificar, raw_2_npy, run_params."
    echo "                   Se não fornecido, o fluxo de trabalho será executado normalmente a partir do início."
    echo ""
    echo "Exemplo de uso: "
    echo "$0 3 'resnet50'"
    echo "$0 3 'resnet50' 'verificar'"
    exit 0
}

# Função para verificar erros após execução de um comando
verificar_erro() {
    if [ $? -ne 0 ]; then
        echo "Erro ao executar $1. O script será encerrado."
        echo ""
        exit 1
    fi
}

# Exibe ajuda se --help ou -h forem fornecidos
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    exibir_ajuda
fi

# Verifica se os parâmetros obrigatórios estão presentes
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Parâmetros 1 e 2 são obrigatórios."
    exibir_ajuda
fi

# Variáveis de entrada
patch=$1
feature=$2
start_point=${3:-"verificar"} # Ponto de partida padrão é "verificar"

# Executa o script Python apropriado baseado no ponto de partida
if [ "$start_point" == "verificar" ]; then
    echo "Executando verificar.py com patch: $patch e feature: $feature"
    python verificar.py $patch $feature
    verificar_erro "verificar.py"
    start_point="raw_2_npy"  # Define o próximo ponto de partida
fi

if [ "$start_point" == "raw_2_npy" ]; then
    echo "Executando raw_2_npy_michel.py com patch: $patch e feature: $feature"
    python raw_2_npy_michel.py $patch $feature
    verificar_erro "raw_2_npy_michel.py"
    start_point="run_params"
fi

if [ "$start_point" == "run_params" ]; then
    for i in $(seq 1 10); do
        echo "Executando run_params_michel.py para a pasta $i com patch: $patch e feature: $feature"
        python run_params_michel.py $i $patch $feature
        verificar_erro "run_params_michel.py para pasta $i"
    done
fi

# Desativa o ambiente virtual
deactivate
