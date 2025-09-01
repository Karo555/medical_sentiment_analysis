# ===== Variables =====
PY ?= python3
export PYTHONPATH := .
PY := python3
CONFIG_BASE := configs/data_base.yaml
CONFIG_LLM  := configs/data_llm.yaml
CONFIG_ENC  := configs/data_encoder.yaml

SAMPLE := data/processed/base/sample.jsonl
ENC_MODE ?= persona_token   # non_personalized | persona_token | personalized_desc
LLM_MODE ?= personalized_desc  # non_personalized | persona_token | personalized_desc | personalized_instruction

CHECKPOINT ?= artifacts/models/encoder/enc_baseline_xlmr
EVAL_SPLIT ?= val

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
		build-llm-pl build-llm-pl-nonpers build-llm-pl-persona-token \
		train-llm-baseline-gemma2 train-llm-persona-token-mistral train-llm-personalized-desc-qwen2 \
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
	@echo ""
	@echo "=== Polish LLM Data ==="
	@echo "build-llm-pl              - materializacja polskiego widoku LLM (personalized_desc)"
	@echo "build-llm-pl-nonpers      - polski widok LLM (non_personalized)"
	@echo "build-llm-pl-persona-token - polski widok LLM (persona_token)"
	@echo ""
	@echo "=== LLM Training ==="
	@echo "train-llm-baseline-gemma2      - train Gemma2-2B baseline (non-personalized)"
	@echo "train-llm-persona-token-mistral - train Mistral-7B with persona tokens"
	@echo "train-llm-personalized-desc-qwen2 - train Qwen2-1.5B with persona descriptions"
	@echo "train-llm-gemma2-27b-personalized - train Gemma2-27B-IT with persona descriptions"
	@echo ""
	@echo "=== LLM Evaluation and Workflow ==="
	@echo "workflow-gemma2-27b-complete   - complete Gemma2-27B workflow (eval + train + viz + compare)"
	@echo "workflow-gemma2-27b-quick      - quick Gemma2-27B workflow (fewer evaluation samples)"
	@echo "eval-llm-gemma2-baseline       - evaluate baseline Gemma2-27B model"
	@echo "eval-llm-gemma2-trained        - evaluate fine-tuned Gemma2-27B model"
	@echo "visualize-llm-training         - visualize training curves (set CHECKPOINT=path)"

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
	$(PY) scripts/prepare_data.py --opinions data/raw/opinions.json --matrix data/raw/emotion_matrix.json --personas-en data/personas/personas_en.json --personas-pl data/personas/personas_pl.json --label_names schema/label_names.json --out data/processed/base/all.jsonl --text_field tekst --default_lang pl

prepare-data-with-id:
	$(PY) scripts/prepare_data.py --opinions data/raw/opinions.json --matrix data/raw/emotion_matrix.json --personas-en data/personas/personas_en.json --personas-pl data/personas/personas_pl.json --label_names schema/label_names.json --out data/processed/base/all.jsonl --text_field tekst --id_field opinion_id --default_lang pl

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

# ===== Polish LLM view materialization =====
build-llm-pl:
	$(PY) scripts/build_llm_view_pl.py --config configs/data_llm_pl.yaml --mode personalized_desc

build-llm-pl-nonpers:
	$(PY) scripts/build_llm_view_pl.py --config configs/data_llm_pl.yaml --mode non_personalized

build-llm-pl-persona-token:
	$(PY) scripts/build_llm_view_pl.py --config configs/data_llm_pl.yaml --mode persona_token

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

train-llm-pllum:
    $(PY) scripts/train_llm.py --config configs/experiment/llm_baseline_pllum.yaml

# ── Eval presets (val domyślnie) ─────────────────────────────────────────────
# Ustaw zmienne środowiskowe, jeśli chcesz inny split/checkpoint.

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

# ── Visualization and Analysis ──────────────────────────────────────────────
visualize-training:
	$(PY) scripts/visualize_training.py --checkpoint $(CHECKPOINT) --label-names schema/label_names.json

analyze-results:
	$(PY) scripts/analyze_results.py --checkpoint $(CHECKPOINT) --split $(EVAL_SPLIT) --label-names schema/label_names.json

visualize-enc-persona-token-xlmr:
	CHECKPOINT=artifacts/models/encoder/enc_persona_token_xlmr \
	$(MAKE) visualize-training

analyze-enc-persona-token-xlmr:
	CHECKPOINT=artifacts/models/encoder/enc_persona_token_xlmr \
	$(MAKE) analyze-results

# ── LLM Training ─────────────────────────────────────────────────────────────
train-llm-baseline-gemma2:
	$(PY) scripts/train_llm.py --config configs/experiment/llm_baseline_gemma2.yaml

train-llm-persona-token-mistral:
	$(PY) scripts/train_llm.py --config configs/experiment/llm_persona_token_mistral.yaml

train-llm-personalized-desc-qwen2:
	$(PY) scripts/train_llm.py --config configs/experiment/llm_personalized_desc_qwen2.yaml

#done
train-llm-gemma2-27b-personalized:
	$(PY) scripts/train_llm.py --config configs/experiment/llm_gemma3_12b_personalized.yaml

# ── LLM Evaluation ──────────────────────────────────────────────────────────
eval-llm-baseline:
	$(PY) scripts/eval_llm.py --config $(CONFIG) --split val --max-samples 100

#done
eval-llm-gemma2-baseline:
	$(PY) scripts/eval_llm.py --config configs/experiment/llm_gemma3_12b_personalized.yaml --split val --output-dir artifacts/models/llm/llm_gemma2_27b_personalized/eval_baseline

eval-llm-gemma2-trained:
	$(PY) scripts/eval_llm.py --config configs/experiment/llm_gemma3_12b_personalized.yaml --split val --checkpoint artifacts/models/llm/llm_gemma2_27b_personalized --output-dir artifacts/models/llm/llm_gemma2_27b_personalized/eval_trained

# ── Complete LLM Workflows ──────────────────────────────────────────────────
workflow-gemma2-27b-complete:
	$(PY) scripts/full_llm_workflow.py --config configs/experiment/llm_gemma3_12b_personalized.yaml

workflow-gemma2-27b-quick:
	PYTHONPATH=/workspace/medical_sentiment_analysis $(PY) scripts/full_llm_workflow.py --config configs/experiment/llm_gemma3_12b_personalized.yaml --quick-eval

visualize-llm-training:
	$(PY) scripts/visualize_llm_training.py --checkpoint $(CHECKPOINT)

# ── Baseline Evaluation (Pre-trained Models) ────────────────────────────────
eval-baseline-xlmr:
	$(PY) scripts/eval_baseline.py --model xlm-roberta-base --split $(EVAL_SPLIT) --num_workers 0

eval-baseline-xlmr-large:
	$(PY) scripts/eval_baseline.py --model xlm-roberta-large --split $(EVAL_SPLIT) --num_workers 0 --eval_bs 2

eval-baseline-mdeberta:
	$(PY) scripts/eval_baseline.py --model microsoft/mdeberta-v3-base --split $(EVAL_SPLIT) --num_workers 0

# ── Model Comparison ──────────────────────────────────────────────────────
compare-xlmr:
	$(PY) scripts/compare_results.py \
		--baseline artifacts/baseline_eval/xlm_roberta_base/baseline_eval_$(EVAL_SPLIT).json \
		--finetuned artifacts/models/encoder/enc_baseline_xlmr/eval_results_$(EVAL_SPLIT).json \
		--model_name "XLM-RoBERTa-base" \
		--output comparison_results/xlmr_base_$(EVAL_SPLIT).json

compare-xlmr-large:
	$(PY) scripts/compare_results.py \
		--baseline artifacts/baseline_eval/xlm_roberta_large/baseline_eval_$(EVAL_SPLIT).json \
		--finetuned artifacts/models/encoder/enc_xlmr_large/eval_results_$(EVAL_SPLIT).json \
		--model_name "XLM-RoBERTa-large" \
		--output comparison_results/xlmr_large_$(EVAL_SPLIT).json

compare-mdeberta:
	$(PY) scripts/compare_results.py \
		--baseline artifacts/baseline_eval/microsoft_mdeberta_v3_base/baseline_eval_$(EVAL_SPLIT).json \
		--finetuned artifacts/models/encoder/enc_mdeberta/eval_results_$(EVAL_SPLIT).json \
		--model_name "mDeBERTa-v3-base" \
		--output comparison_results/mdeberta_v3_base_$(EVAL_SPLIT).json
