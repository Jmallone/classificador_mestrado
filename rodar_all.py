import os
import subprocess
from datetime import datetime
import logging
import send_discord as sd

# Configuração do logging
logging.basicConfig(filename="execution.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Diretório onde estão os arquivos .txt
DIR_PATH = "features_raw/"
LOG_DIR = "logs/"

def extrair_parametros(file_name):
    """
    Extrai os parâmetros a partir do nome do arquivo .txt.

    O nome do arquivo tem o formato: EFFICIENTNETB0FMD_1X1.txt.
    Esta função separa o nome em dois parâmetros:
    1. O parâmetro numérico final (ex: '1' de '1X1')
    2. O nome do modelo (ex: 'efficientnetb0fmd')

    Args:
        file_name (str): O nome do arquivo .txt.

    Returns:
        tuple: Um par de strings contendo o parâmetro numérico e o nome do modelo.
    """
    logging.info(f"Processando arquivo: {file_name}")
    # O formato dos arquivos é algo como EFFICIENTNETB0FMD_1X1.txt
    parts = file_name.split('_')
    
    # Extraindo o parâmetro 1X1 ou 2X2
    first_param = parts[-1].replace('.txt', '').split('X')[-1]
    
    # Extraindo o restante do nome do arquivo (ex: EFFICIENTNETB0FMD -> efficientnetb0fmd)
    second_param = parts[0].lower()

    return first_param, second_param

def rodar_comando(first_param, second_param):
    """
    Executa um comando bash baseado nos parâmetros extraídos do arquivo .txt.

    Cria um arquivo de log para armazenar a saída do comando. O comando é executado
    de forma síncrona, ou seja, a função espera sua conclusão antes de continuar.

    Args:
        first_param (str): O parâmetro numérico extraído do nome do arquivo.
        second_param (str): O nome do modelo extraído do nome do arquivo.

    Raises:
        subprocess.CalledProcessError: Caso ocorra um erro na execução do comando.
    """
    try:
        # Obtém a data e hora atual no formato desejado
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        # Cria o nome do arquivo de log com a data
        log_file = os.path.join(LOG_DIR, f"{current_time}_{first_param}_{second_param}.log")
        
        # Verifica se o diretório de logs existe, caso contrário, cria-o
        os.makedirs(LOG_DIR, exist_ok=True)

        # Comando a ser executado
        command = f"./roda_unitario_michel.sh {first_param} '{second_param}' > {log_file} 2>&1"
        
        logging.info(f"Rodando comando: {command}")
        
        # Executa o comando bash e aguarda a conclusão
        subprocess.run(command, shell=True, check=True)
        
        logging.info(f"Comando finalizado com sucesso: {command}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar o comando: {e}")

def processar_arquivos(dir_path):
    """
    Itera sobre todos os arquivos .txt em um diretório e processa cada um deles.

    Para cada arquivo .txt encontrado no diretório, a função extrai os parâmetros
    do nome do arquivo e executa o comando correspondente.

    Args:
        dir_path (str): O caminho do diretório onde os arquivos .txt estão localizados.
    """
    # Encontra todos os arquivos .txt
    txt_files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
    
    # Itera sobre os arquivos encontrados
    for file_name in txt_files:
        # Extrai os parâmetros do nome do arquivo
        first_param, second_param = extrair_parametros(file_name)

        # Instancia a classe com o webhook do Discord
        notificacao = sd.ResultadosNotificacao()
        notificacao.enviar_mensagem_generica(f'✅ Executando {file_name} Parametros {first_param}')
        
        # Executa o comando com os parâmetros extraídos
        rodar_comando(first_param, second_param)

if __name__ == "__main__":
    """
    Ponto de entrada do script.

    Verifica se o diretório de arquivos .txt existe. Se sim, inicia o processamento dos arquivos.
    Caso contrário, registra um erro informando que o diretório não foi encontrado.
    """
    # Verifica se o diretório existe
    if os.path.exists(DIR_PATH):
        processar_arquivos(DIR_PATH)
    else:
        logging.error(f"Diretório não encontrado: {DIR_PATH}")
