

import numpy as np
import glob
import requests

class ResultadosNotificacao:
    
    def __init__(self, webhook_url='https://discord.com/api/webhooks/1159986861532520510/nsfqOyvwUeFGIJK0HQa-c0nX7IIjz6eb1eZx4wmlJjL6NEAhseZEBKvTgXZbe7v595jy', output_file='resultados.txt'):
        self.webhook_url = webhook_url
        self.output_file = output_file
        self.resultados = {}

    def calcular_estatisticas(self, resultados):
        return np.mean(resultados), np.std(resultados), np.min(resultados), np.max(resultados)

    def imprimir_estatisticas(self, resultados):
        media, desvio, mini, maxi = self.calcular_estatisticas(resultados)
        return " %.2f +- %.2f, min: %.2f, max: %.2f\n" % (media, desvio, mini, maxi)

    def faz_resultados(self):
        # Redirecionar a saída para um arquivo
        with open(self.output_file, 'w') as output_file:
            patchs = glob.glob('resultados/*')
            patchs.sort()

            for p in patchs:
                try:
                    titulo = p.split('resultados/')[-1].replace('_', ' ').upper()
                    print('------------------- ' + titulo.rjust(10), end=' ---------------')
                    print()
                    output_file.write('------------------- ' + titulo.rjust(10) + ' ---------------\n')
                    
                    # Inicializar dicionário de resultados
                    self.resultados = {
                        'knn': [], 'rf': [], 'svm': [], 'mlp': [],
                        'svm_pca=16': [], 'knn_pca=16': [], 'rf_pca=16': [], 'mlp_pca=16': [],
                        'svm_pca=32': [], 'knn_pca=32': [], 'rf_pca=32': [], 'mlp_pca=32': [],
                        'svm_pca=64': [], 'knn_pca=64': [], 'rf_pca=64': [], 'mlp_pca=64': [],
                    }

                    # Carregar os resultados
                    for i in range(1, 11):
                        tmp = np.load(p + '/test' + str(i) + '_result.npy', allow_pickle=True).item()
                        for k in tmp.keys():
                            self.resultados[k].append(tmp[k])

                    # Filtrar apenas os resultados que contêm dados
                    self.resultados = {k: v for k, v in self.resultados.items() if v}

                    # Mostrar e salvar os Resultados
                    for teste in sorted(self.resultados):
                        print(teste.rjust(10), end=' - ')
                        res = self.imprimir_estatisticas(self.resultados[teste])
                        output_file.write('`' + res.replace('\n', '') + ' - ' + teste.rjust(10) + '`\n')
                    print()
                    output_file.write('\n')
                except Exception as e:
                    print(f"Erro ao processar {p}: {e}")
                    continue

    def enviar_notificacao(self):
        print("\n---------------- NOTIFICAÇÃO -----------------")
        print("+-------------- ---- Descrição ---- -------------+")
        print("| Envia a Notificação dos resultados.            |")
        print("+-------------- ---- --------- ---- -------------+")
        try:
            # Processar os resultados
            self.faz_resultados()

            # Ler o conteúdo do arquivo gerado
            with open(self.output_file, 'r') as file:
                texto = file.read()

            # Enviar cada seção dos resultados como uma notificação separada
            for t in texto.split('-------------------'):
                payload = {
                    "content": '### --------------- ' + t,
                    "username": 'Servidor Diego UTFPR',
                    "avatar_url": 'https://assets.turbologo.com/blog/en/2021/07/22032835/Firefox_logo-2017.png'
                }
                requests.post(self.webhook_url, data=payload)

        except Exception as e:
            print(f'Erro ao enviar notificação: {e}')
    
    # Função para enviar uma mensagem genérica
    def enviar_mensagem_generica(self, mensagem):
        print("\n---------------- NOTIFICAÇÃO -----------------")
        print("+-------------- ---- Descrição ---- -------------+")
        print("| Envia a Notificação dos resultados.            |")
        print("+-------------- ---- --------- ---- -------------+")

        try:
            payload = {
                "content": mensagem,
                "username": 'Servidor Diego UTFPR',
                "avatar_url": 'https://assets.turbologo.com/blog/en/2021/07/22032835/Firefox_logo-2017.png'
            }
            response = requests.post(self.webhook_url, data=payload)
            if response.status_code == 204:
                print("Mensagem enviada com sucesso!")
            else:
                print(f"Falha ao enviar mensagem. Status code: {response.status_code}")
        except Exception as e:
            print(f'Erro ao enviar mensagem: {e}')

