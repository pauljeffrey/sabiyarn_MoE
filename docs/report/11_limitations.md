## 11. Limitations, risks, licences

**Scientific**
1. *No downstream result yet.* Everything about quality of the 306M model rests on validation loss (§7.5, §10.2).
2. *Upcycling + pruning provenance is missing* (Appendix D): the experiments that would show the MoE beats the dense model at equal training FLOPs are not in the repository.
3. *Absolute learned positions* cap the context at 32,768, cost 8.2% of the parameters, and 87% of the table (rows ≥ 4,096) has never been trained (§3.7, §9.3). A comparison with RoPE is outside the current plan.
4. *The 12 new languages' tags are first embedded in SFT* (§7.6): during continued pretraining they are learned from text only.
5. *Repetition:* languages from Hausa down are seen four times (§6.4); memorisation and a rising train/validation gap are expected in the smallest languages.
6. *Synthetic and machine-translated data:* LLM-written data are admitted only with a gating score; Aya-type collections carry machine-translation artefacts (94% of `aya_collection` QA targets); the QE audit is a sample (≈30 rows per group), has a failure line calibrated on one hand-checked set, and is blind to adequacy errors QE does not catch; QE itself is unreliable for four language pairs and absent for Ibibio/Urhobo.
7. *Language identification* is the weakest link of the restructuring: GlotLID + stop-words are reliable for long texts and unreliable for short or code-switched ones; the hedged `unsure` form and the family prior mitigate but do not remove label noise in `<|input_lang|>`.
8. *Reward hacking* of a learned metric (AfriCOMET-QE) is a known risk; the design (calibrated anchors, flat top, chrF++ anchor option, copy guard, scaffold gate) reduces it but there is no human evaluation yet.

**Data licences and provenance (must be resolved before releasing weights or data)**
* The authors' rule was *no non-commercial licence*; one deliberate exception was made on request: `xlsum` rows inside `naija`/`naija2` (CC-BY-NC-SA) are kept and flagged `license_flag`. Decide: exclude, or release the model as research-only, or ask for permission.
* Watchtower/JW text is explicitly admitted by the authors; its redistribution terms differ from its reading terms. Keep it in training data only, do not republish it.
* Aya collection and public benchmark-derived sets keep their own licences (several Apache/CC-BY, some CC-BY-SA); the combined `sft-all` repository is **private** and should stay so until a per-source licence table exists (`docs/report/generated/sft_all_stats.md` lists source names).
* LLM-generated data carries the provider's terms (the generator model names are in the data-gen manifests).
* Personal data: web-scraped text has not been through a PII scrub; none is documented in the repository.

**Safety.** No safety/harm evaluation exists. The SFT sets contain a hate-speech classification set (`hate`) whose *inputs* are offensive by construction; they are classification inputs, but a safety review of the data and the model is needed before deployment.

**Compute & environmental note.** Throughput numbers are in §4; a total GPU-hour count for the project is not recorded (Appendix D, item 14).
