## 10. Evaluation

### 10.1 What exists **[V]**
* **Validation loss on fixed, stratified blocks** (§6.7) — the only signal that has been run on the 306M model.
* **`eval_suite/`** (`python -m eval_suite.run`): translation English↔X (FLORES-200 devtest, else MAFAND-MT; chrF++ headline, chrF, BLEU; optional AfriCOMET), topic classification (SIB-200, MasakhaNEWS), sentiment (AfriSenti), NER (MasakhaNER 2.0, entity-level F1), MMLU (AfriMMLU + English; 5-shot; likelihood-scored). Languages with a standard benchmark: yor, hau, ibo, twi, ewe, fon, pcm (+ English); Efik, Urhobo, Fulah have none and report `n/a`. Prompts exist in two styles: `tag` (the v1 multitask format, for base checkpoints) and `chat` (the SFT template).
* **`eval/eval.py`** — the SabiYarn-125M evaluation; **not comparable** with the suite (NER by per-sentence label presence; topics by keyword rules).
* **AfriCOMET-QE** — reference-free, used for data audit and RL reward (§8.5, §9.4).
* **Start-up generation probes** and the hand-written curated probe set (`eval` loss on a probe file; the YAML warns "if it is far below train/val loss, suspect the bins, not the model").

### 10.2 What is missing for a paper **[A]**
1. No benchmark number for the 306M model is stored in the repository (Appendix D, item 1).
2. The eval suite covers the original languages; it has no task data for kin, sna, zul, swh, orm, wol, xho, tsn, nya, afr, som, sot. Candidate benchmarks: FLORES+ (translation, all of them), SIB-200 (topic, all of them), AfriMMLU / Global-MMLU where available, MasakhaNEWS, MasakhaNER 2.0 (kin, sna, zul, swh, xho, wol, nya …), AfriSenti (swh, orm, kin …), Belebele (reading comprehension), AfriXNLI. The MMLU "train" splits were used as SFT data (MMMLU-Yoruba and multilingual MMLU dev) — **dev/test separation must be kept** when these benchmarks are reported.
3. Decontamination between the bins/SFT sets and every evaluation set (hash n-gram overlap against FLORES+, SIB-200 and the MMLU test splits).
4. Comparable baselines at the same budget: the dense 125M, a 306M dense with the same active parameters, and public multilingual models of similar size (e.g. mT5-small/base, AfroXLMR, Gemma-3 270M / SmolLM2 360M where their language coverage permits), using the same prompts.
5. Metrics for the assistant-turn grammar itself: exact-match of the scaffold (input-language set, plan verbs, target language), calibration of the hedged `unsure` form (accuracy when unsure vs when certain), and tool-call validity.
6. Efficiency measurements in the paper's format: tokens/s and MFU at the *final* configuration on 4 × A100, and active-vs-total FLOP comparison against the dense baseline at equal quality.
