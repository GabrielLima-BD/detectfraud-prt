# DetectFraud-prt

Projeto de detecção de fraudes em transações desenvolvido para um desafio técnico, utilizando um conjunto de dados disponibilizado pela organização do desafio (no formato típico de transações + identidade). O foco aqui é entregar um pipeline completo: preparação dos dados, treino supervisionado, seleção de limiar (threshold), avaliação, predição em lote e disponibilização via API e dashboard.

## Objetivo

- Estimar a probabilidade de fraude por transação.
- Controlar o trade-off entre recall e precisão via threshold configurável.
- Entregar um fluxo reprodutível (config central + artefatos de modelo) e pontos de consumo (FastAPI/Streamlit).

## Principais componentes

- Processamento e merge dos dados de transação/identidade por chave (ex.: `TransactionID`).
- Treino supervisionado com CatBoost (bom desempenho em dados tabulares e suporte a variáveis categóricas).
- Seleção automática de threshold baseada em métrica configurável (padrão: F1).
- Relatórios em `reports/` (métricas e gráficos) e artefatos em `models/`.
- Consumo do modelo:
	- API (FastAPI) para predição via JSON.
	- Dashboard (Streamlit) para upload de CSV e download do resultado.

## Estrutura do projeto

```
.
├── data/
│   ├── raw/                      # Dados brutos 
│   └── processed/                # Dados processados para treino/predição
├── models/                       # Modelo e artefatos (ex.: catboost_model.cbm, artifacts.json)
├── reports/                      # Métricas e gráficos gerados no treino
├── src/
│   ├── config/config.yaml        # Configuração central do pipeline
│   ├── data/load_merge_ieee.py   # Merge e preparação
│   ├── models/                   # Treino/avaliação/predição
│   └── utils/                    # Métricas e utilitários
├── api_fastapi.py                # API de predição
├── app_streamlit.py              # Dashboard interativo
├── pred_template.csv             # Template de colunas para predição
├── requirements.txt
└── Makefile
```

## Requisitos

- Python 3.10+ (recomendado)

## Instalação

Crie e ative um ambiente virtual:

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dados

Por padrão, o pipeline espera os arquivos brutos em `data/raw/` com os nomes definidos em `src/config/config.yaml` (ex.: `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`).

## Como executar

### 1) Processar/merge dos dados

```bash
python -m src.data.load_merge_ieee --config src/config/config.yaml
```

Saída esperada (configurável):

- `data/processed/train_merged.parquet`
- `data/processed/test_merged.parquet`
- Resumos JSON com colunas/categorias

### 2) Treinar o modelo supervisionado

```bash
python -m src.models.train_supervised --config src/config/config.yaml
```

Artefatos gerados:

- `models/catboost_model.cbm`
- `models/artifacts.json` (features, colunas categóricas, threshold escolhido, métricas, parâmetros)
- `reports/metrics.json`
- Imagens em `reports/` (matriz de confusão, curva Precisão-Recall, SHAP)

### 3) Predição em lote (CSV)

Use `pred_template.csv` como referência de cabeçalhos.

```bash
python -m src.models.predict --config src/config/config.yaml --input caminho/para/entrada.csv --output saida.csv
```

O resultado inclui:

- `fraud_proba`: probabilidade estimada
- `fraud_flag`: classificação binária usando o threshold salvo em `models/artifacts.json`

## Dashboard (Streamlit)

```bash
streamlit run app_streamlit.py
```

O dashboard permite:

- Upload de CSV
- Ajuste de threshold em tempo real
- Download do CSV com `fraud_proba` e `fraud_flag`

## API (FastAPI)

Suba a API em modo desenvolvimento:

```bash
uvicorn api_fastapi:app --reload
```

Endpoints:

- `GET /health` → status
- `POST /predict` → predição

Formato do payload:

```json
{
	"registros": [
		{"coluna1": 123, "coluna2": "abc"},
		{"coluna1": 456, "coluna2": "def"}
	]
}
```

## Atalhos via Makefile

```bash
make setup
make process
make train
make eval
make predict INPUT=caminho/para/arquivo.csv OUTPUT=pred.csv
make app
make api
```

No Windows, se `make` não estiver disponível, rode os comandos Python diretamente.

## Configuração e reprodutibilidade

Os principais parâmetros do pipeline ficam em `src/config/config.yaml`:

- Caminhos de entrada/saída
- Estratégias de preprocessamento (ex.: preenchimento de missing, downcast)
- Hiperparâmetros do CatBoost
- Métrica para seleção de threshold
- Configuração de explicabilidade (SHAP)

## Observações

- Este repositório não deve conter dados sensíveis ou identificáveis. Use sempre dados anonimizados/permitidos pela organização do desafio.

## Troubleshooting rápido

| Sintoma | Possível causa | Como resolver |
| --- | --- | --- |
| `ModuleNotFoundError` | Não instalou dependências | Rode `pip install -r requirements.txt` no ambiente ativo |
| Arquivo Parquet não salva | PyArrow não instalado | Já está no requirements, mas se deu erro, instale manualmente `pip install pyarrow` ou desabilite Parquet no config |
| Modelo reclama de colunas | CSV veio com cabeçalho diferente | Gere template a partir de `pred_template.csv` e garanta mesma ordem |
| Streamlit/uvicorn não iniciam | Porta em uso | Rode com `--server.port 8502` (Streamlit) ou `--port 8001` (Uvicorn) |
| Métricas fracas | Dados sujos ou leak | Revise colunas vazias, crie features novas, ajuste threshold |
