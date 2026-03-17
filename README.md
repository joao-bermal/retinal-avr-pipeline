<h1 align="center">
  👁️ AVR Cardiovascular Risk Analysis Pipeline
</h1>

<p align="center">
  Segmentação avançada de vasos retinianos e classificação Arteríola/Vênula utilizando <i>Deep Learning</i> para estimativa de risco cardiovascular (TCC).
</p>

---

## 📌 Introdução

Este projeto consiste em um repositório Python Modular (Codebase) focado no cálculo da relação AVR (Arteriolar-to-Venular Ratio) através da análise de retinografias. Originalmente baseado em pesados *Jupyter Notebooks*, a arquitetura foi completamente refatorada para uma esteira lógica escalável e rastreável.

A solução é composta por duas redes neurais primárias:
1. **Modelagem de Segmentação Vascular** (`EnhancedUNet`): Responsável por isolar a árvore de vasos do fundo do olho em relação à textura original. (Atingindos >0.79 Dice Score sobre o formato DRIVE).
2. **Modelagem de Classificação A/V** (`EnhancedMultiDatasetAVNet`): Classifica ramificações segmentadas, identificando artérias e veias.

---

## 🛠️ Arquitetura do Codebase (`src/`)

Para propósitos de **Engenharia de Machine Learning (MLOps)** e reprodutibilidade científica, abolimos a declaração desorganizada em notebooks e consolidamos o motor analítico da seguinte forma:

```text
├── data/                  # Root para as bases DRIVE, IOSTAR, RITE (Baixe o zip no Google Drive link na seção de Execução)
├── EXECUTION_GUIDE.md     # 🚀 Manual passo a passo de como rodar do Zero 
├── main.py                # 🚪 Ponto Único de Entrada (Entrypoint/CLI)
├── notebooks/
│   └── 00_Unified_Master_Pipeline.ipynb  # 📊 Caderno final e unificado com gráficos & inferência
├── src/
│   ├── config/            # Variáveis globais de ambiente e Thresholds 
│   ├── data/              # Extração PyTorch (Dataset classes) e pre-processing scripts 
│   ├── loss/              # (Consolidado: src/training/losses.py) DICE / BCD / Focal
│   ├── models/            # Arquiteturas Neurais: U-Net, MultiDatasetAVNet 
│   ├── pipeline/          # Wrapper que une predições de ponta-a-ponta (ScientificAVRPipeline)
│   └── training/          # Loop Otimizado (Epochs, val metrics, persistência de checkpoints)
└── tests/
    └── sanity_check.py    # Teste rápido contra Memory Leaks ou falha arquitetural (Dummy-pass)
```

---

## ⚙️ Instalação e Requisitos

Este projeto depende ativamente de tensores montados via GPU e processamento OpenCV.

```bash
# Clone e ative o ambiente Python
git clone https://github.com/joao-bermal/retinal-avr-pipeline.git
cd retinal-avr-pipeline

# Instale os requerimentos do seu requirements.txt compatível com CUDA
pip install -r requirements.txt
```

---

## 🚀 Como Executar?

Não há necessidade de tocar nas configurações profundas se você apenas deseja rodar com hiperparâmetros canônicos. Por favor, leia o nosso **[Guia de Execução (EXECUTION_GUIDE.md)](./EXECUTION_GUIDE.md)** para instruções completas abrindo ponta a ponta.

**Resumo de comandos CLI (`main.py`)**:

- Treinar U-Net de segmentação vascular:
  `python main.py --train_seg`
- Treinar Multi-Net de Classificação A/V:
  `python main.py --train_av`
- Executar e exibir a pipeline completa sobre uma imagem:
  `python main.py --run_pipeline "data/drive/test/images/minha_imagem.tif"`
- Treinar alterando parâmetros (ex: aumentar epochs e taxa de erro):
  `python main.py --train_seg --epochs 200 --lr 0.001`

Para análises médicas, relatórios gráficos iterativos ou defesas de TCC utilizando `Matplotlib`, recomendamos que acione diretamente nosso Jupyter Notebook Mestre:
> `notebooks/00_Unified_Master_Pipeline.ipynb`
