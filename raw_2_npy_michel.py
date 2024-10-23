import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from joblib import delayed, Parallel
from scipy.stats import ttest_ind_from_stats
import sys
from glob import glob
print("\n------------------ RAW_2_NPY -------------------")
print("+-------------- ---- Descrição ---- -------------+")
print("| Transformando o arquivo passado por parâmetro  |")
print("| em formato .npy para futura análise.           |")
print("+-------------- ---- --------- ---- -------------+")

try:
# VARIAVELS GLOBAIS
    patch = sys.argv[1]
    nome_modelo = sys.argv[2].upper()

    fname = f'features_raw/{nome_modelo.upper()}_{patch}X{patch}.txt'
    folder = f'{nome_modelo.lower()}_{patch}x{patch}/' # Criar essa pasta em /features/
    features_raw = glob('features_raw/*')
    labels = glob('labels/*')
    pasta = folder.replace('/', '')


    print("--- Configuração Inicial")
    print(f"fname: {fname}")
    print(f"folder: {folder}")
    print(f"pasta: {pasta}")
    print("-------------------------\n")
except Exception as e:
    print("\n+-------ERRO-----------------------------------")
    print("|", str(e))
    print("| Erro ao ler os argumentos")
    print("+----------------------------------------------")
    sys.exit(1)

def load_data_shape(filename):
#     print('--- Load Data Shape')
    with open(filename) as f:
        c = f.readline()
    c = [i for i in c.strip().split(" ")]
#     print('--- FIM - Load Data Shape')
    return len(c)

def load_data(filename, class_idx, features_idxs):
    print("- ** Carregando Dados")
    with open(filename) as f:
        lines = f.readlines()
#     print("------ Load Data")

    # Criar uma matriz de zeros com as dimensões corretas
    num_rows = len(lines)
    num_cols = len(lines[0].strip().split(" "))-1
    c = np.zeros((num_rows, num_cols), dtype=float)
    F = []

    # Preencher a matriz linha por linha
    for i, line in enumerate(lines):
        dados = line.strip().split(" ")
        c[i] = np.array(dados[:-1], dtype=float)
        F.append(dados[-1])
        
    X = c[:, features_idxs].astype(float)
    Y = c[:, class_idx].astype(int).astype(str)

    print('Y: {}'.format(Y))
#     print("------ Fim Load Data")
    return X, Y, F

try:
    X, Y, F = load_data(fname, -1, slice(0,load_data_shape(fname)-2))
    F = np.array([i.split('/')[-1] for i in F])
    Y = Y.astype(int)

    for l in labels:
        l_name = l.replace('labels/','').replace('.txt','')

        with open(l) as f:
            c = f.readlines()
        
        c = np.array([np.array(i.strip().split(" ")) for i in c])
        c = np.array([i[0].split('/')[-1] for i in c])

        idx = np.where(np.isin(F, c))[0]
        

        test_x = X[idx].astype(float)
        test_y = Y[idx].astype(int)
        test_f = F[idx]

        X_OUT = 'features/'+folder+l_name+'_X.npy'
        Y_OUT = 'features/'+folder+l_name+'_Y.npy'
        F_OUT = 'features/'+folder+l_name+'_F.npy'

        np.save(X_OUT, test_x)
        np.save(Y_OUT, test_y)
        np.save(F_OUT, test_f)
        
        print("+---------------------------------------------")
        print('| Feito a extração de: '+l_name+' com sucesso!')
    print("+---------------------------------------------\n")
    for tipo in ['train', 'test', 'val']:
        for i in range(1,11):
            

            X_name = f"features/{pasta}/{tipo}{i}_X.npy"
            Y_name = f"features/{pasta}/{tipo}{i}_Y.npy"
            F_name = f"features/{pasta}/{tipo}{i}_F.npy"
            

            x = np.load(X_name)
            y = np.load(Y_name)
            F = np.load(F_name, allow_pickle=True)

            labels = pd.DataFrame(F, columns=['file'])

            X = pd.DataFrame(x)
            Y = pd.DataFrame(y)

            only_labels = labels['file'].value_counts().keys()

            new_X = pd.DataFrame()
            new_Y = pd.DataFrame()
            new_F = pd.DataFrame()

            for label in only_labels:
                indices = labels[labels['file'] == label].index

                new_F = pd.concat([new_F, labels.loc[indices]])
                new_X = pd.concat([new_X, X.loc[indices]])
                new_Y = pd.concat([new_Y, Y.loc[indices]])
            print("+---------------------------------------------")
            print("| Salvando: "+str(X_name))
            print("+---------------------------------------------")
            np.save(X_name, new_X)
            np.save(Y_name, new_Y)
            np.save(F_name, new_F)
except Exception as e:
    print("\n+-------ERRO-----------------------------------")
    print("| "+str(e))
    print("| Erro ao ler os dados")
    print("+----------------------------------------------")
    sys.exit(1)