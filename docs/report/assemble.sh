#!/bin/bash
# Concatenate the report parts (and the generated tables, if present) into docs/TECHNICAL_REPORT.md
set -e
cd "$(dirname "$0")"
out=../TECHNICAL_REPORT.md
: > "$out"
for f in 00_front.md 01_lineage.md 02_tokenizer.md 03_architecture.md 04_efficiency.md 05_data_v1.md 06_data_campaign.md 07_training.md 08_post_training_data.md 09_post_training_methods.md 10_evaluation.md 11_limitations.md 12_paper_plan.md 13_appendices.md; do
  cat "$f" >> "$out"; printf '\n\n' >> "$out"
done
for g in generated/*.md; do
  [ -e "$g" ] && { printf '\n\n' >> "$out"; cat "$g" >> "$out"; }
done
echo "wrote $out ($(wc -w < "$out") words)"
