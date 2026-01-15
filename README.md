# Projeto Baby Fraud Detector 👶🕵️

Bem-vindo ao seu projeto de detecção de fraudes IEEE-CIS explicado como se você fosse um bebê curioso. Cada etapa diz o que faz, por que faz e como não tropeçar.

## Visão geral simples

1. **Dados**: baixamos os arquivos do Kaggle e juntamos `train_transaction` com `train_identity` usando `TransactionID`. Treino e teste ficam em `data/processed/` prontinhos.
2. **Pré-processamento**: preenchemos valores ausentes (números viram 0, categorias viram "missing"), reduzimos o tamanho dos números (downcast) e salvamos em Parquet para economizar memória.
3. **Modelo supervisionado**: CatBoost aprende com as transações rotuladas (`isFraud`) usando pesos automáticos para lidar com o desbalanceamento (muito mais transações normais do que fraudulentas).
4. **Threshold inteligente**: em vez do padrão 0.5, testamos vários limiares e escolhemos o que dá melhor F1 na curva Precisão-Recall.
5. **Avaliação**: medimos Precisão, Recall, F1, ROC AUC e PR AUC. Geramos matriz de confusão e curva PR para enxergar os trade-offs.
6. **Predição**: um script confere se o novo CSV tem as mesmas colunas e devolve `fraud_proba` e `fraud_flag`.
7. **Dashboard**: Streamlit permite subir um CSV, ajustar o threshold com slider e baixar os resultados.
8. **API**: FastAPI recebe registros via JSON e devolve probabilidades.
9. **Explainability**: calculamos SHAP em uma amostra pequena para entender quais colunas empurram a decisão.

## Estrutura de pastas

```
├── data/
│   ├── raw/                # CSVs originais do Kaggle (depois do download)
│   └── processed/          # Arquivos prontos para treino/predição (Parquet ou CSV fallback)
├── models/                 # catboost_model.cbm + artifacts.json + modelos extras
├── reports/                # Métricas em JSON e gráficos (confusão, PR, SHAP)
├── src/
│   ├── config/config.yaml  # Parâmetros centrais do projeto
│   ├── data/load_merge_ieee.py
│   ├── models/
│   │   ├── train_supervised.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── train_unsupervised.py
│   └── utils/metrics.py
├── app_streamlit.py
├── api_fastapi.py
├── pred_template.csv       # Cabeçalhos corretos para predição
├── requirements.txt
├── Makefile
├── .env.example
└── README.md
```

## Setup passo a passo (super detalhado)

### 1. Clonar o projeto

```bash
# Linux/macOS
git clone <url-do-repo>
cd DetectFraud-prod
```

```powershell
# Windows PowerShell
git clone <url-do-repo>
cd DetectFraud-prod
```

### 2. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

```powershell
pip install -r requirements.txt
```

### 4. Configurar Kaggle CLI

1. Crie token em https://www.kaggle.com/account (botão "Create New API Token").
2. Copie `.env.example` para `.env` e preencha `KAGGLE_USERNAME` e `KAGGLE_KEY`.
3. Execute:

```bash
export $(cat .env | xargs)
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip data/raw/ieee-fraud-detection.zip -d data/raw
```

```powershell
Get-Content .env | ForEach-Object { if ($_ -match "^") { $name,$value = $_.Split('='); Set-Item -Path Env:$name -Value $value } }
kaggle competitions download -c ieee-fraud-detection -p data/raw
Expand-Archive data/raw/ieee-fraud-detection.zip -DestinationPath data/raw
```

### 5. Processar dados

```bash
python -m src.data.load_merge_ieee --config src/config/config.yaml
```

### 6. Treinar modelo supervisionado

```bash
python -m src.models.train_supervised --config src/config/config.yaml
```

Isso gera:
- `models/catboost_model.cbm`
- `models/artifacts.json` (features, threshold, métricas)
- Gráficos em `reports/`

### 7. Avaliar (opcional, se tiver conjunto de validação rotulado)

```bash
python -m src.models.evaluate --config src/config/config.yaml --dataset data/processed/train_merged.parquet
```

### 8. Predizer em novo CSV

```bash
python -m src.models.predict --config src/config/config.yaml --input caminho/para/novo.csv --output saida.csv
```

Use `pred_template.csv` para garantir o cabeçalho correto.

### 9. Abrir dashboard Streamlit

```bash
streamlit run app_streamlit.py
```

### 10. Subir API FastAPI (modo desenvolvimento)

```bash
uvicorn api_fastapi:app --reload
```

Endpoint principal: `POST /predict` com JSON no formato `{ "registros": [ {...}, {...} ] }`

### 11. Modelo não supervisionado (opcional)

```bash
python -m src.models.train_unsupervised --config src/config/config.yaml
```

## Explicações de bebê

### Por que o dataset é desbalanceado?
A base IEEE-CIS tem **muito mais transações normais** do que fraudes. É assim na vida real: só uma partezinha é fraude. Se treinarmos sem cuidado, o modelo aprende a dizer "não é fraude" sempre e parece ter alta acurácia, mas isso é inútil. Usamos `class_weights` automáticos no CatBoost para dar mais peso às fraudes. Outra opção seria SMOTE, mas ele cria dados sintéticos e pode ficar pesado aqui. Comece com class weights (mais simples e estável) e só teste SMOTE se precisar.

### Por que PR AUC e Recall importam mais que Acurácia?
- **Recall** diz quantas fraudes reais você achou. Se for baixo, o banco perde dinheiro.
- **Precisão** diz quantos dos alertas eram fraudes de verdade.
- **PR AUC** (área sob curva Precisão-Recall) combina os dois e é robusta em datasets desbalanceados. Acurácia pode enganar: acertar 99% das transações normais não significa detectar fraudes.

### Problemas comuns e soluções
- **Memória insuficiente**: use Parquet (menor) e downcast (números ocupam menos). Se travar, processe em chunks (pode adaptar `load_merge_ieee.py`).
- **Sem credenciais do Kaggle**: sem arquivo `kaggle.json` configurado (ou `.env`), o download falha. Veja seção de setup.
- **Colunas divergentes**: se o CSV novo não tem todas as colunas do treino, `predict.py` avisa exatamente quais faltam. Preencha ou remova com as mesmas transformações do preprocessamento.

### O que funciona bem
- **CatBoost** lida com categorias grandes sem precisar de one-hot (evita explosão de memória).
- **Threshold customizado** mantém o trade-off calibrado para seu negócio.
- **Métricas certas (PR AUC, Recall)** mostram desempenho real em fraudes.

### O que evitar
- **One-hot massivo**: IEEE-CIS tem centenas de colunas, one-hot vira desastre de memória.
- **Acurácia como métrica principal**: não mede fraude direito.
- **Threshold 0.5 fixo**: quase nunca otimiza Recall em datasets desbalanceados. Sempre olhe a curva PR.

## Explainability (SHAP)
Após o treino, olhe `reports/shap_summary.png`. Ele mostra as colunas que mais empurram a probabilidade de fraude (cores vermelhas ou azuis indicando valores altos/baixos). Use isso para explicar aos times de risco por que certas transações foram marcadas.

## Makefile: atalhos úteis

```bash
make process        # roda merge + preprocessamento
make train          # treina CatBoost
make eval           # reavalia com dados rotulados
make predict INPUT=data/processed/test_merged.parquet OUTPUT=pred.csv
make app            # inicia Streamlit
make api            # inicia FastAPI
```

No Windows, use `make` via Git Bash ou rode os comandos Python equivalentes manualmente.

## Checklist pós-instalação

- [ ] Ambiente virtual criado e `pip install -r requirements.txt` ok.
- [ ] Dados do Kaggle baixados em `data/raw/`.
- [ ] `python -m src.data.load_merge_ieee` executado sem erros.
- [ ] Arquivos `models/catboost_model.cbm` e `models/artifacts.json` gerados.
- [ ] Gráficos em `reports/` (confusão, PR, SHAP) disponíveis.
- [ ] `pred_template.csv` usado como referência para novos dados.
- [ ] Dashboard `streamlit run app_streamlit.py` abre sem reclamações.
- [ ] API `uvicorn api_fastapi:app --reload` responde `GET /health` com `{ "status": "ok" }`.

## Troubleshooting rápido

| Sintoma | Possível causa | Como resolver |
| --- | --- | --- |
| `ModuleNotFoundError` | Não instalou dependências | Rode `pip install -r requirements.txt` no ambiente ativo |
| Arquivo Parquet não salva | PyArrow não instalado | Já está no requirements, mas se deu erro, instale manualmente `pip install pyarrow` ou desabilite Parquet no config |
| Modelo reclama de colunas | CSV veio com cabeçalho diferente | Gere template a partir de `pred_template.csv` e garanta mesma ordem |
| Streamlit/uvicorn não iniciam | Porta em uso | Rode com `--server.port 8502` (Streamlit) ou `--port 8001` (Uvicorn) |
| Métricas fracas | Dados sujos ou leak | Revise colunas vazias, crie features novas, ajuste threshold |

Pronto! Agora você tem um pipeline completo, pensado para ser leve, explicável e fácil de rodar, com todos os alertas de armadilhas comuns. Bons experimentos! 🧪✨
