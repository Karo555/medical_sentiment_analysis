preview-llm:
	python scripts/preview_llm_view.py --config configs/data_llm.yaml --sample data/processed/base/sample.jsonl --mode personalized_desc

preview-enc:
	python scripts/preview_encoder_view.py --config configs/data_encoder.yaml --sample data/processed/base/sample.jsonl --mode persona_token

preview-enc-tokenize:
	python scripts/preview_encoder_view.py --config configs/data_encoder.yaml --sample data/processed/base/sample.jsonl --mode persona_token --tokenize