# 🎯 Projeto de Mestrado

Tem como objetivo a classificação a partir dos conjuntos de dados **DTD** e **FMD**, assim como a avaliação para diferentes tipos de _backbones_, a exemplo, resnet, vgg, inception entre outros.

### 🔧 Dependências

As dependências do projeto estão listadas no arquivo requirements.txt. Para instalar as dependências, execute:

```bash
pip install -r requirements.txt
```

### 🚀  Passos para Executar

- 🔍 **Verifique o _dataset_:** Confirme se está utilizando DTD ou FMD. Neste exemplo, vamos utilizar DTD.

- 📤 **Inserir as _features_:** Navegue até a pasta `features_raw/DTD_DIEGO` e coloque as _features_ extraídas lá.

- ⚙️ **Converter as _features_:** Rode o notebook `converter.ipynb` para realizar a conversão das _features_.

- 📂 **Substituir os _labels_:** Volte para a pasta raiz e substitua a pasta _labels_ pela pasta de _labels_ correspondente ao _dataset_ utilizado. Neste caso, execute:

```bash
cp -r labels_DTD labels
```

- 📊 **Executar o _pipeline_:** Após realizar as etapas anteriores, execute o _script_ principal para rodar todo o _pipeline_ de treinamento:

```bash
nohup python rodar_all.py &> saida.log &
```

## ℹ️ Observação sobre o Nome dos Arquivos

Os arquivos de _features_ seguem o seguinte padrão de nomenclatura:

`[DATASET][FEATURE][PATCH x PATCH].txt`

- **[DATASET]**: Nome do conjunto de dados (ex: `DTD`, `FMD`).
- **[FEATURE]**: Nome da feature extraída (ex: `obif`, `resnet`).
- **[PATCH x PATCH]**: Tamanho dos patches utilizados (ex: `1x1`, `2x2`).

Exemplo: `DTD_obif_1x1.txt`

## ℹ️ Observação sobre o Conteúdo das Features

Cada arquivo de _features_ deve conter, no mínimo, as seguintes colunas:

`[FEATURES_NUMBERS] [NOME_LABEL] [NOME_FILE]`

- **[FEATURES_NUMBERS]**: Valores numéricos das _features_ extraídas.
- **[NOME_LABEL]**: Nome ou valor correspondente ao label da imagem.
- **[NOME_FILE]**: Nome do arquivo associado àquela entrada de _feature_.

Exemplo de linha: `00000 0.000000 0.000000 banded banded_0004.jpg`
