"""Evaluation suite for SabiYarn: translation (chrF++/BLEU), topic classification, sentiment,
NER and MMLU, on the standard benchmarks that cover the model's languages.

    python -m eval_suite.run --model <checkpoint dir or HF id> --tasks all --langs all --limit 200

See HOW_TO_RUN.md ("Evaluating a checkpoint").
"""
