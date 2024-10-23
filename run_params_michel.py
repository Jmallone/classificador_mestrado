import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from joblib import delayed, Parallel
from scipy.stats import ttest_ind_from_stats
import pickle
from glob import glob
import send_discord as sd
import sys
from time import time

def salvar_modelo(nome: str, modelo, patch: int) -> None:
    """
    Salva o modelo treinado em um arquivo .pkl.

    Args:
        nome (str): Nome do modelo.
        modelo (object): Modelo treinado a ser salvo.
        patch (int): Número de patches por lado.
    """
    diretorio_modelos = "modelos"
    os.makedirs(diretorio_modelos, exist_ok=True)  # Cria diretório se não existir

    caminho_modelo = os.path.join(diretorio_modelos, f"{nome}_best_ad_{patch}.pkl")
    
    try:
        with open(caminho_modelo, "wb") as f:
            pickle.dump(modelo, f)
        print(f"Modelo salvo com sucesso: {caminho_modelo}")
    except Exception as e:
        print(f"Erro ao salvar o modelo {nome}: {e}")

def get_final_predictions_maj(n_patches: int, predicted: np.array) -> np.array:
    """
    Calcula as previsões finais agrupando por patches e selecionando a previsão 
    mais frequente para cada grupo.

    Args:
        n_patches (int): Número de patches por lado (quadrado).
        predicted (np.array): Array de previsões.

    Returns:
        np.array: Array com as previsões finais por grupo.
    """
    n_patches_total = n_patches * n_patches
    songs = predicted.shape[0] // n_patches_total
    reshaped_preds = predicted.reshape((songs, n_patches_total))
    majority_preds = [np.bincount(row).argmax() for row in reshaped_preds]
    return np.array(majority_preds)

def get_final_predictions_sum(n_patches, predicted):
    n_patches = n_patches*n_patches
    songs = int(predicted.shape[0] / n_patches)

    print(predicted.shape[0], n_patches)
    # print(songs, n_patches)
    r = predicted.reshape((songs,n_patches))
    r = [np.unique(i, return_counts=True) for i in r]
    r = [ i[0][np.argmax(i[1])] for i in r]
    
    return r


def get_labels(n_patches: int, labels: np.array) -> np.array:
    """
    Seleciona os rótulos baseados nos patches agrupados.

    Args:
        n_patches (int): Número de patches por lado.
        labels (np.array): Array com os rótulos.

    Returns:
        np.array: Array com os rótulos agrupados.
    """
    n_patches_total = n_patches * n_patches
    return labels[::n_patches_total]

def selecionar_melhor_modelo(classificador, X_treino, X_val, y_treino, y_val, n_jobs=4, 
                             cv_folds=None, params={}):
    

    # Configura o score de acordo com o número de classes
    score_fn = 'f1' if len(set(y_treino)) < 3 else 'f1_weighted'

    # Inicializa o contador de tempo
    start_time = time()

    # GridSearchCV com o callback
    clf = GridSearchCV(classificador(), params, cv=cv_folds, n_jobs=n_jobs, 
                        scoring=score_fn, verbose=0)
    
    # Ajuste do modelo
    clf.fit(np.vstack((X_treino, X_val)), np.hstack((y_treino, y_val)))
    
    # Melhor combinação de parâmetros e o melhor score
    melhor_comb = clf.best_params_
    melhor_val = clf.best_score_

    elapsed_time = (time() - start_time) / 60  # tempo em minutos
    print(f"Tempo de execução: {elapsed_time:.2f} minutos")

    return clf, melhor_comb, melhor_val

def do_cv(nome: str, classificador, X_teste: np.array, y_teste: np.array, X_treino: np.array, y_treino: np.array, 
          X_val: np.array, y_val: np.array, cv_splits: int, param_cv_folds: int = None, n_jobs: int = 8, 
          scale: bool = False, dim_red=None, params: dict = {}, patch: int = 1) -> dict:
    """
    Realiza a validação cruzada, treinamento e avaliação do modelo.

    Args:
        nome (str): Nome do modelo.
        classificador (callable): Função que retorna um modelo a ser treinado.
        X_teste (np.array): Dados de teste.
        y_teste (np.array): Rótulos de teste.
        X_treino (np.array): Dados de treino.
        y_treino (np.array): Rótulos de treino.
        X_val (np.array): Dados de validação.
        y_val (np.array): Rótulos de validação.
        cv_splits (int): Número de splits para validação cruzada.
        param_cv_folds (int, optional): Número de folds para validação cruzada.
        n_jobs (int): Número de jobs paralelos para o treinamento.
        scale (bool, optional): Indica se os dados devem ser normalizados.
        dim_red (callable, optional): Função de redução de dimensionalidade (ex: PCA).
        params (dict): Dicionário de parâmetros para o modelo.
        patch (int): Número de patches por lado.

    Returns:
        dict: Resultados com scores e tempos de treino/teste.
    """
    resultados = {}

    # Pré-processamento (normalização e redução de dimensionalidade)
    if scale:
        scaler = StandardScaler()
        X_treino = scaler.fit_transform(X_treino)
        X_val = scaler.transform(X_val)
        X_teste = scaler.transform(X_teste)
    
    if dim_red:
        dr_model = dim_red()
        X_treino = dr_model.fit_transform(X_treino)
        X_val = dr_model.transform(X_val)
        X_teste = dr_model.transform(X_teste)

    print(f"Treinando o modelo {nome}...")

    # Treinamento e seleção de melhores parâmetros
    modelo_treinado, melhores_param, melhor_f1_val = selecionar_melhor_modelo(
        classificador, X_treino, X_val, y_treino, y_val, n_jobs, param_cv_folds, params
    )

    print(f"Melhores parâmetros para {nome}: {melhores_param}")
    print(f"Melhor F1-score na validação: {melhor_f1_val:.4f}")

    # Avaliação no conjunto de teste
    pred_teste = modelo_treinado.predict(X_teste)
    y_teste_final = get_labels(patch, y_teste)
    pred_final = get_final_predictions_sum(patch, pred_teste)
    f1_teste = f1_score(y_teste_final, pred_final, average='weighted' if len(set(y_teste)) > 2 else 'binary')

    # Resultados
    resultados['f1_val'] = melhor_f1_val
    resultados['f1_teste'] = f1_teste
    resultados['best_params'] = melhores_param

    # Salvamento do modelo
    salvar_modelo(nome, modelo_treinado, patch)

    return resultados

def run(output_proto, pasta, patch):
    """
    Executa o pipeline de treinamento, avaliação e salvamento dos resultados 
    para diferentes classificadores e patches.

    Args:
        output_proto (list): Lista com os arquivos de saída (test, train, val).
        pasta (str): Nome da pasta onde os resultados serão salvos.
        patch (int): Número de patches por lado.

    Returns:
        dict: Resultados por número de patches.
    """
    resultados_per_patches = {}
    
    from sklearn.svm import SVC
    
    # Parâmetros do classificador SVM
    svm_params = {
        'C': [0.001, 0.1, 1, 10, 100, 1000], 
        'gamma': ['auto', 'scale'],
        'kernel': ['rbf']
    }

    # Carregando os arquivos de features e labels
    TEST_X = f"features/{output_proto[0]}_X.npy"
    TEST_Y = f"features/{output_proto[0]}_Y.npy"
    TRAIN_X = f"features/{output_proto[1]}_X.npy"
    TRAIN_Y = f"features/{output_proto[1]}_Y.npy"
    VAL_X = f"features/{output_proto[2]}_X.npy"
    VAL_Y = f"features/{output_proto[2]}_Y.npy"

    test_X = np.load(TEST_X)
    test_y = np.load(TEST_Y).reshape(-1)
    train_X = np.load(TRAIN_X)
    train_y = np.load(TRAIN_Y).reshape(-1)
    val_X = np.load(VAL_X)
    val_y = np.load(VAL_Y).reshape(-1)

    print(f"X_TRAIN: \n  {train_X[0][:20]}")
    print(f"Y_TRAIN:   {train_y}")
    print(f"X_TRAIN_FILE:   {TRAIN_X}")
    print("\n")
    
    # Lista de classificadores a serem testados (apenas SVM neste caso)
    classifiers = [
        ('svm', SVC, False, svm_params),
    ]

    resultados = {}
    melhor_f1_geral = -1  # Variável para armazenar o melhor F1-score geral
    melhor_modelo_geral = None  # Variável para armazenar o melhor modelo

    for nome, classificador, scale, params in classifiers:
        try:
            # Executa a validação cruzada e o treinamento do modelo
            r = do_cv(nome, classificador, test_X, test_y, train_X, train_y, val_X, val_y, 
                      cv_splits=10, param_cv_folds=10, n_jobs=os.cpu_count()-1, scale=scale, params=params, patch=patch)
            
            resultados[nome] = r["f1_teste"]
            
            # Verificar se este é o melhor modelo até agora
            if r["f1_teste"] > melhor_f1_geral:
                melhor_f1_geral = r["f1_teste"]
                melhor_modelo_geral = r["best_params"]
                
        except Exception as e:
            print(f"Erro ao treinar o modelo {nome}: {e}")
            continue

    # Salva os resultados em um arquivo .npy
    file_name = f"resultados/{output_proto[0]}_result.npy"
    np.save(file_name, resultados)

    # Salvar o melhor modelo geral ao final
    if melhor_modelo_geral is not None:
        salvar_modelo(f"melhor_modelo_geral_{pasta}", melhor_modelo_geral, patch)

    resultados_per_patches[patch] = resultados

    return resultados_per_patches

# Argumentos recebidos da linha de comando
i = sys.argv[1] 
patch = sys.argv[2]
pasta = sys.argv[3]

# Formatação da pasta para incluir o número de patches
pasta = f'{pasta}_{patch}x{patch}'

# Exibição de cabeçalho e parâmetros
print("\n---------------- RODAR MODELOS -----------------")
print("+-------------- ---- Descrição ---- -------------+")
print("| Rodando varios modelos com os arquivos .npy    |")
print("| e salvando seus resultados.                    |")
print("+-------------- ---- --------- ---- -------------+")

print("---------- Parâmetros ----------")
print(f"i: {i}")
print(f"patch: {patch}")
print(f"pasta: {pasta}")
print("---------- ---------- ----------")

# Instancia a classe com o webhook do Discord
notificacao = sd.ResultadosNotificacao()

# Iniciando o processo
print('Iniciando: '+str(i)+'  --   '+str(pasta))
notificacao.enviar_mensagem_generica(f'Iniciando [i] - {str(i)}')
# Inicializa o contador de tempo
start_time = time()

try:
    # Monta os nomes dos arquivos de saída (test, train, val)
    output_proto = [
        f'{pasta}/test{int(i)}',
        f'{pasta}/train{int(i)}',
        f'{pasta}/val{int(i)}'
    ]
    
    # Executa o pipeline de treinamento e avaliação
    resultado_final = run(output_proto, pasta, int(patch))
    
    # Exibe os resultados finais
    print(resultado_final)
    print(f'Feito a extração de: {i} com sucesso!')

    notificacao.enviar_mensagem_generica(f'resultado_final {str(resultado_final)}')
    elapsed_time = (time() - start_time) / 60  # tempo em minutos
    notificacao.enviar_mensagem_generica(f"Tempo de execução: {elapsed_time:.2f} minutos")
    notificacao.enviar_mensagem_generica(f"\n ----------- \n")

except Exception as e:
    # Em caso de erro, exibe a mensagem e envia uma notificação via webhook
    print(f'Erro ao extrair: {i}')
    
    # Instancia a classe com o webhook do Discord
    notificacao = sd.ResultadosNotificacao()
    notificacao.enviar_mensagem_generica(f'Erro ao extrair: i[{str(i)}] patch[{str(patch)}] pasta[{str(pasta)}]')
    notificacao.enviar_mensagem_generica(str(e))
    
    # Também exibe o erro no console
    print(e)