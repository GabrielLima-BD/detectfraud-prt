PYTHON?=python
CONFIG?=src/config/config.yaml
RAW_DIR?=data/raw
PROCESSED_DIR?=data/processed
MODEL?=models/catboost_model.cbm
ARTIFACTS?=models/artifacts.json

.PHONY: setup download process train eval predict app api clean

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

download:
	@echo "Baixe os dados executando:"
	@echo "  kaggle competitions download -c ieee-fraud-detection -p $(RAW_DIR)"
	@echo "Depois descompacte os arquivos na pasta data/raw."

process:
	$(PYTHON) -m src.data.load_merge_ieee --config $(CONFIG)

train:
	$(PYTHON) -m src.models.train_supervised --config $(CONFIG)

unsupervised:
	$(PYTHON) -m src.models.train_unsupervised --config $(CONFIG)

.eval:
	$(PYTHON) -m src.models.evaluate --config $(CONFIG)

eval: .eval

predict:
	@if [ -z "$(INPUT)" ]; then \
		echo "Use INPUT=/caminho/para/arquivo.csv make predict"; \
		exit 1; \
	fi
	$(PYTHON) -m src.models.predict --config $(CONFIG) --input $(INPUT) --output $(OUTPUT)

app:
	streamlit run app_streamlit.py

api:
	uvicorn api_fastapi:app --reload

clean:
	rm -f $(MODEL) $(ARTIFACTS) reports/*.png reports/*.json models/*.joblib
