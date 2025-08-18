preview-llm:
	python scripts/preview_llm_view.py --config configs/data_llm.yaml --sample data/processed/base/sample.jsonl --mode personalized_desc

preview-enc:
	python scripts/preview_encoder_view.py --config configs/data_encoder.yaml --sample data/processed/base/sample.jsonl --mode persona_token

preview-enc-tokenize:
	python scripts/preview_encoder_view.py --config configs/data_encoder.yaml --sample data/processed/base/sample.jsonl --mode persona_token --tokenize

preview-base:
	python scripts/preview_base_data.py --config configs/data_base.yaml --sample data/processed/base/sample.jsonl

prepare-data:
	python scripts/prepare_data.py --opinions data/raw/opinions.json --matrix data/raw/emotion_matrix.json --personas data/personas/personas.json --label_names schema/label_names.json --out data/processed/base/all.jsonl --text_field tekst --default_lang pl

prepare-data-with-id:
	python scripts/prepare_data.py --opinions data/raw/opinions.json --matrix data/raw/emotion_matrix.json --personas data/personas/personas.json --label_names schema/label_names.json --out data/processed/base/all.jsonl --text_field tekst --id_field opinion_id --default_lang pl

make-splits:
	python scripts/make_splits.py --config configs/splits.yaml