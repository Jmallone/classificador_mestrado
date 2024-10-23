import sys
import glob
import os

def criar_pasta(pasta):
    try:
        if os.path.exists(pasta) and os.path.isdir(pasta):
            print("Pasta encontrada:", pasta)
        else:
            print("Pasta não encontrada:", pasta)
            print("Criando nova pasta...")
            os.makedirs(pasta)
            print("Nova pasta criada:", pasta)
    except OSError as e:
        print("\n+-------ERRO-----------------------------------")
        print("| Erro ao criar pasta:", pasta)
        print("+----------------------------------------------")
        sys.exit(1)

if len(sys.argv) < 3:
    print("Parâmetros 1 e 2 são obrigatórios.")
    print("Exemplo de uso: ")
    print("python verificar.py <patch> <modelo>")
    print("python verificar.py 5 resnet50")
    sys.exit(1)

patch = sys.argv[1]
modelo = sys.argv[2].upper()

print("\n-------------- VERIFICANDO PASTAS --------------")
print("+-------------- ---- Descrição ---- -------------+")
print("| Verificando se o arquivo passado por parâmetro |")
print("| existe, e se suas pastas tambem, caso contrario|")
print("| será criado pastas para ele.                   |")
print("+-------------- ---- --------- ---- -------------+")


arquivo_modelo = f"{modelo}_{patch}X{patch}"

arquivos_encontrados = glob.glob(f"features_raw/{arquivo_modelo}.txt")
print(arquivos_encontrados)

if len(arquivos_encontrados) > 0:
    print("Arquivo(s) encontrado(s):")
    for arquivo in arquivos_encontrados:
        print(arquivo)
else:
    print("\n+-------ERRO-----------------------------------")
    print("| Nenhum arquivo encontrado com o nome:", arquivo_modelo)
    print("+----------------------------------------------")
    sys.exit(1)

pasta_modelo = f"features/{arquivo_modelo.lower()}"
criar_pasta(pasta_modelo)

pasta_resultados = f"resultados/{arquivo_modelo.lower()}"
criar_pasta(pasta_resultados)

pasta_resultados = f"modelos/{arquivo_modelo.lower()}"
criar_pasta(pasta_resultados)
