# 🚀 Guia de Execução End-to-End (Do Zero ao Pipeline)

Este guia descreve o passo a passo completo para rodar o projeto desde o **Passo 0** (sem nenhum modelo treinado) até a geração de *métricas, inferências e predições integradas*. Toda a arquitetura foi modularizada em Python nativo (`src/`) para permitir execuções em terminais e automações MLOps, além de centralizar as visualizações na pasta `notebooks/`.

---

## 📦 Passo 0: Preparação Inicial

### 1. Ambiente e Dependências
Certifique-se de que você está num ambiente virtual ativo (venv ou conda) com as bibliotecas essenciais instaladas, em especial o PyTorch.

```bash
pip install -r requirements.txt
```

*(Nota: Caso utilize uma GPU NVIDIA, verifique se a versão instalada do `torch` e `torchvision` correspondem ao seu CUDA Toolkit).*

### 2. Disposição dos Dados (Datasets)
Como as imagens pesam >130MB, elas não ficam no GitHub. Você precisa fazer o download do arquivo `data.zip` hospedado externamente pelo autor:
1. **Baixe o arquivo `data.zip` em:** [Google Drive - Retinal AVR Data](https://drive.google.com/file/d/1VUNfJkRd8V9RmR--NlnI_RZg84Ioz-E8/view?usp=sharing)
2. **Extraia o conteúdo** diretamente na raiz do projeto, para que a pasta `data/` seja recriada com as subpastas esperadas pelo sistema.

Exemplo da estrutura resultante para o treino de Segmentação com o DRIVE:

```text
data/
└── drive/
    ├── training/
    │   ├── images/
    │   └── 1st_manual/
    └── test/
        ├── images/
        └── 1st_manual/
```

*(Faça o mesmo para os datasets de classificação A/V - IOSTAR, RITE, LES-AV).*

---

## 🏋️‍♂️ Passo 1: Treinamento dos Modelos (Codebase)

Todo o treinamento dos modelos ocorre através do nosso Entry Point central: o `main.py`. Ele consome as funções de treinamento (`src/training/`) e passa por todo o split, carregamento e logs, utilizando os hyperparâmetros de `src/config/settings.py`.

### A) Treinar o Modelo de Segmentação (Enhanced U-Net)
Para iniciar o treinamento da rede neural responsável por extrair a árvore vascular das imagens (Segmentação):

```bash
python main.py --train_seg
```
- **O que acontece**: O dataloader processará as imagens `*.tif` aplicando transformações geométricas e de cor (CLAHE), invocará a `EnhancedUNet`, e salvará o arquivo de peso `.pth` na pasta destino mapeada (ex: `models/segmentation/best_model.pth`).
- **Logs e Métricas**: Os scores de Loss e a evolução (Dice loss, epochs, etc) serão impressos rodada a rodada no terminal.

### B) Treinar o Modelo de Classificação A/V
Para iniciar o processamento da arquitetura complexa A/V:

```bash
python main.py --train_av
```
- **O que acontece**: Ele utilizará os datasets indicados no `settings.py` para alimentar o `EnhancedMultiDatasetAVNet`.
- O peso final será salvo (ex: `models/av_classification/best_avnet.pth`).

### C) Ajuste Dinâmico de Hiperparâmetros (Opcional)
Você notará no `main.py` que não é mais preciso editar o código-fonte manualmente para trocar parâmetros corriqueiros de treinamento. Basta usar as tags CLI:

```bash
python main.py --train_seg --epochs 250 --batch_size 8 --lr 0.0001
```

Opcionalmente, crie um arquivo `experimento_1.yaml` para sobrescrever variáveis complexas de `settings.py` e rode:
```bash
python main.py --train_av --config experimento_1.yaml
```

---

## 📊 Passo 2: Geração de Estatísticas e Testes Isolados

Caso você queira atestar o *forward/backward* ou realizar um "Dry-Run" nos modelos com tensores mockados para averiguação de sintaxe de máquina/memória da GPU sem carregar todo o arquivo de dados:

```bash
python tests/sanity_check.py
```

Esse script isolado garante que a arquitetura dos PyTorch Models (`src/models/*`) compila matematicamente com matrizes preenchidas zero/um antes de comprometer horas de processamento.

---

## 🧠 Passo 3: Avaliação e Unified Pipeline (Inferência e Gráficos)

Uma vez que as pastas sob `models/` possuam os pesos treinados `.pth`, o pipeline está livre para realizar predições do mundo real, empilhar predições sobrepostas e plotar gráficos. 

Este passo pode ser feito individualmente por imagem via terminal:
```bash
python main.py --run_pipeline "data/drive/test/images/01_test.tif"
```

### Visualizações Interativas, Relatórios e Gráficos (Via Jupyter Notebook)
A análise profunda dos resultados foi transferida com exclusividade e controle visual unificado para o notebook mestre:

1. Inicie o Servidor Jupyter:
   ```bash
   jupyter notebook notebooks/00_Unified_Master_Pipeline.ipynb
   ```
2. **Dentro do Notebook:**
   - Execute a Célula 1 & 2 para invocar o Backend Modular `ScientificAVRPipeline()`.
   - Você pode passar lotes numéricos de teste ou caminhos de imagens diretos.
   - O objeto Pipeline retornará com dicionários detalhando o tempo de inferência (`inference_time_ms`), a matriz de erro (`mask`), e permitirá plotar relatórios médicos colorizados lado-a-lado usando `matplotlib`. Todo o *spaghetti code* de treino está isolado longe da sua visão iterativa.
3. Este é o ambiente principal para você expor na sua Defesa do TCC, gerando relatórios precisos do modelo treinado através do motor Python.
