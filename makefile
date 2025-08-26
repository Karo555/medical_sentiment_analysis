# ===== Variables =====
PY ?= python
export PYTHONPATH := .
PY := python
CONFIG_BASE := configs/data_base.yaml
CONFIG_LLM  := configs/data_llm.yaml
CONFIG_ENC  := configs/data_encoder.yaml

SAMPLE := data/processed/base/sample.jsonl
ENC_MODE ?= persona_token   # non_personalized | persona_token | personalized_desc
LLM_MODE ?= personalized_desc  # non_personalized | persona_token | personalized_desc | personalized_instruction

CHECKPOINT ?= artifacts/models/encoder/enc_baseline_xlmr
EVAL_SPLIT ?= val  # val | test

CALIB_LABELS ?=
TOPK_ECE ?=
CALIB_BINS ?= 15
CALIB_STRATEGY ?= uniform
CALIB_DPI ?= 140

# ===== Phony =====
.PHONY: help preview-base preview-llm preview-enc preview-enc-tokenize \
        prepare-data prepare-data-with-id make-splits \
        build-enc-view build-enc-nonpers build-enc-persona-token build-enc-personalized \
		build-llm-view build-llm-nonpers build-llm-persona-token build-llm-personalized-desc build-llm-personalized-instruction \
		eval-enc eval-enc-test

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

train-enc-baseline:
	$(PY) scripts/train_encoder.py --config configs/experiment/enc_baseline.yaml

eval-enc:
	$(PY) scripts/eval_encoder.py --config configs/experiment/enc_baseline.yaml --split $(EVAL_SPLIT) --checkpoint $(CHECKPOINT)

eval-enc-test:
	$(PY) scripts/eval_encoder.py --config configs/experiment/enc_baseline.yaml --split test --checkpoint $(CHECKPOINT)

calib-enc:
	$(PY) scripts/plot_calibration.py \
		--checkpoint $(CHECKPOINT) \
		--split $(EVAL_SPLIT) \
		--label-names schema/label_names.json \
		--bins $(CALIB_BINS) \
		--strategy $(CALIB_STRATEGY) \
		$(if $(CALIB_LABELS),--labels $(CALIB_LABELS),) \
		$(if $(TOPK_ECE),--topk-ece $(TOPK_ECE),) \
		--dpi $(CALIB_DPI)

check-splits:
	$(PY) scripts/check_splits.py --all data/processed/base/all.jsonl --splits-dir data/splits

# ── Train presets ────────────────────────────────────────────────────────────
train-enc-persona-token-xlmr:
	$(PY) scripts/train_encoder.py --config configs/experiment/enc_persona_token_xlmr.yaml

train-enc-personalized-xlmr:
	$(PY) scripts/train_encoder.py --config configs/experiment/enc_personalized_desc_xlmr.yaml

train-enc-persona-token-mdeberta:
	$(PY) scripts/train_encoder.py --config configs/experiment/enc_persona_token_mdeberta.yaml

train-enc-personalized-mdeberta:
	$(PY) scripts/train_encoder.py --config configs/experiment/enc_personalized_desc_mdeberta.yaml

# ── Eval presets (val domyślnie) ─────────────────────────────────────────────
# Ustaw zmienne środowiskowe, jeśli chcesz inny split/checkpoint.
EVAL_SPLIT ?= val

eval-enc-persona-token-xlmr:
	CHECKPOINT=artifacts/models/encoder/enc_persona_token_xlmr \
	$(PY) scripts/eval_encoder.py --config configs/experiment/enc_persona_token_xlmr.yaml --split $(EVAL_SPLIT) --checkpoint $$CHECKPOINT

eval-enc-personalized-xlmr:
	CHECKPOINT=artifacts/models/encoder/enc_personalized_desc_xlmr \
	$(PY) scripts/eval_encoder.py --config configs/experiment/enc_personalized_desc_xlmr.yaml --split $(EVAL_SPLIT) --checkpoint $$CHECKPOINT

eval-enc-persona-token-mdeberta:
	CHECKPOINT=artifacts/models/encoder/enc_persona_token_mdeberta \
	$(PY) scripts/eval_encoder.py --config configs/experiment/enc_persona_token_mdeberta.yaml --split $(EVAL_SPLIT) --checkpoint $$CHECKPOINT

eval-enc-personalized-mdeberta:
	CHECKPOINT=artifacts/models/encoder/enc_personalized_desc_mdeberta \
	$(PY) scripts/eval_encoder.py --config configs/experiment/enc_personalized_desc_mdeberta.yaml --split $(EVAL_SPLIT) --checkpoint $$CHECKPOINT

# ── Calibration (opcjonalnie) ────────────────────────────────────────────────
calib-enc-persona-token-xlmr:
	CHECKPOINT=artifacts/models/encoder/enc_persona_token_xlmr \
	$(MAKE) calib-enc

calib-enc-personalized-xlmr:
	CHECKPOINT=artifacts/models/encoder/enc_personalized_desc_xlmr \
	$(MAKE) calib-enc

calib-enc-persona-token-mdeberta:
	CHECKPOINT=artifacts/models/encoder/enc_persona_token_mdeberta \
	$(MAKE) calib-enc

calib-enc-personalized-mdeberta:
	CHECKPOINT=artifacts/models/encoder/enc_personalized_desc_mdeberta \
	$(MAKE) calib-enc
