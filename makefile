# ===== Variables =====
PY := python
CONFIG_BASE := configs/data_base.yaml
CONFIG_LLM  := configs/data_llm.yaml
CONFIG_ENC  := configs/data_encoder.yaml

SAMPLE := data/processed/base/sample.jsonl
ENC_MODE ?= persona_token   # non_personalized | persona_token | personalized_desc
LLM_MODE ?= personalized_desc  # non_personalized | persona_token | personalized_desc | personalized_instruction

# ===== Phony =====
.PHONY: help preview-base preview-llm preview-enc preview-enc-tokenize \
        prepare-data prepare-data-with-id make-splits \
        build-enc-view build-enc-nonpers build-enc-persona-token build-enc-personalized \
		build-llm-view build-llm-nonpers build-llm-persona-token build-llm-personalized-desc build-llm-personalized-instruction

help:
	@echo "preview-base              - smoke test kontraktu bazowego"
	@echo "preview-llm               - podglad promptu LLM (domyślnie personalized_desc)"
	@echo "preview-enc               - podglad wejścia encodera (ENC_MODE=$(ENC_MODE))"
	@echo "preview-enc-tokenize      - j.w. + tokenizacja"
	@echo "prepare-data              - preprocess -> all.jsonl (bez ID w wejściu)"
	@echo "prepare-data-with-id      - preprocess -> all.jsonl (z ID w wejściu)"
	@echo "make-splits               - generuj splity wg configs/splits.yaml"
	@echo "build-enc-view            - materializacja widoku encodera (ENC_MODE=$(ENC_MODE))"
	@echo "build-enc-nonpers         - widok encodera (non_personalized)"
	@echo "build-enc-persona-token   - widok encodera (persona_token)"
	@echo "build-enc-personalized    - widok encodera (personalized_desc)"

# ===== Previews / smoke tests =====
preview-base:
	$(PY) scripts/preview_base_data.py --config $(CONFIG_BASE) --sample $(SAMPLE)

preview-llm:
	$(PY) scripts/preview_llm_view.py --config $(CONFIG_LLM) --sample $(SAMPLE) --mode personalized_desc

preview-enc:
	$(PY) scripts/preview_encoder_view.py --config $(CONFIG_ENC) --sample $(SAMPLE) --mode $(ENC_MODE)

preview-enc-tokenize:
	$(PY) scripts/preview_encoder_view.py --config $(CONFIG_ENC) --sample $(SAMPLE) --mode $(ENC_MODE) --tokenize

# ===== Data prep & splits =====
prepare-data:
	$(PY) scripts/prepare_data.py --opinions data/raw/opinions.json --matrix data/raw/emotion_matrix.json --personas data/personas/personas.json --label_names schema/label_names.json --out data/processed/base/all.jsonl --text_field tekst --default_lang pl

prepare-data-with-id:
	$(PY) scripts/prepare_data.py --opinions data/raw/opinions.json --matrix data/raw/emotion_matrix.json --personas data/personas/personas.json --label_names schema/label_names.json --out data/processed/base/all.jsonl --text_field tekst --id_field opinion_id --default_lang pl

make-splits:
	$(PY) scripts/make_splits.py --config configs/splits.yaml

# ===== Encoder view materialization =====
build-enc-view:
	$(PY) scripts/build_encoder_view.py --config $(CONFIG_ENC) --mode $(ENC_MODE)

build-enc-nonpers:
	$(PY) scripts/build_encoder_view.py --config $(CONFIG_ENC) --mode non_personalized

build-enc-persona-token:
	$(PY) scripts/build_encoder_view.py --config $(CONFIG_ENC) --mode persona_token

build-enc-personalized:
	$(PY) scripts/build_encoder_view.py --config $(CONFIG_ENC) --mode personalized_desc

# ===== llm view materialization =====
build-llm-view:
	$(PY) scripts/build_llm_view.py --config $(CONFIG_LLM) --mode $(LLM_MODE)

build-llm-nonpers:
	$(PY) scripts/build_llm_view.py --config $(CONFIG_LLM) --mode non_personalized

build-llm-persona-token:
	$(PY) scripts/build_llm_view.py --config $(CONFIG_LLM) --mode persona_token

build-llm-personalized-desc:
	$(PY) scripts/build_llm_view.py --config $(CONFIG_LLM) --mode personalized_desc

build-llm-personalized-instr:
	$(PY) scripts/build_llm_view.py --config $(CONFIG_LLM) --mode personalized_instruction

register-tokens:
	python -m modules.data.tokenizer_utils --config configs/tok.yaml
