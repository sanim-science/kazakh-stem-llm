# Does fine-tuning on new Kazakh STEM questions improve KazMMLU?

I wrote a set of 2,000 Kazakh school STEM multiple-choice questions, fine-tuned two open language models on them, and measured the models on the held-out part of my set and on [KazMMLU](https://huggingface.co/datasets/MBZUAI/KazMMLU), the main Kazakh benchmark.

**Result:** both models learned my questions (+12 and +18 points on 413 held-out questions) but **neither improved on KazMMLU**. This held across three training runs per model and several checks for leakage and answer-letter bias.

![Change in accuracy after fine-tuning, Llama and Qwen](two_model_change.png)

| Mean of 3 training seeds | Llama-3.1-8B-Instruct | Qwen2.5-7B-Instruct |
|---|---|---|
| My 413 held-out questions | 43.3% → 55.7% (**+12.3**, 95% CI +7.9 to +16.9) | 43.3% → 61.4% (**+18.1**, +13.2 to +22.8) |
| KazMMLU Kazakh STEM, 4,174 questions | 40.8% → 41.3% (+0.6, −0.8 to +2.0) | 47.0% → 46.2% (−0.8, −2.4 to +0.8) |
| KazMMLU Kazakh non-STEM, 2,045 questions | 42.1% → 42.0% (−0.1) | 36.9% → 37.5% (+0.6) |

Intervals are 95% bootstrap intervals over questions (10,000 resamples). Adding seed-to-seed variation (resampling seeds and questions) widens them slightly; the conclusions do not change (`robustness_checks.txt`).

## Links

- Dataset: [Sanim09/kazakh-stem-mcq-2000](https://huggingface.co/datasets/Sanim09/kazakh-stem-mcq-2000) (CC BY 4.0)
- Adapters: [Sanim09/Llama-3.1-8B-Instruct-KazSTEM-LoRA](https://huggingface.co/Sanim09/Llama-3.1-8B-Instruct-KazSTEM-LoRA), [Sanim09/Qwen2.5-7B-Instruct-KazSTEM-LoRA](https://huggingface.co/Sanim09/Qwen2.5-7B-Instruct-KazSTEM-LoRA)

## Findings

1. **Gains on my questions, none on KazMMLU, in both models.** On my held-out questions the gain is significant for every training seed (exact McNemar p < 0.0001). On KazMMLU only one of the six runs reached p < 0.05 (Llama seed 2, +1.7 points, p = 0.04, uncorrected for the number of tests), the other seeds of the same model gave −0.2 and +0.2, and the three-seed means are not significant for either model.
2. **The held-out gain is not driven by near-copies.** The 338 held-out questions least similar to any training question (similarity below 0.6) still gain +10.8 (Llama) and +16.2 (Qwen).
3. **KazMMLU questions that resemble training questions do not gain significantly more** than the rest, in either model.
4. **The small KazMMLU movements come from a letter habit.** After fine-tuning both models answer "D" too often on KazMMLU's 4-option questions (my questions have 5 options with a balanced key). Removing the habit without using the answer key leaves KazMMLU unchanged: Llama 41.7% → 41.4%, Qwen 47.2% → 46.8%.
5. **Biology gains least** on my questions in both models (+2.9 and +4.0), while informatics, math and physics gain 11 to 32 points.
6. **Llama's one KazMMLU hint (chemistry +3.4) did not replicate on Qwen (−0.2)** and is not significant after correcting for five subjects.
7. **The evaluation is reproducible:** re-running Qwen's starting score gave identical numbers in every subject.

## Method

- **Data.** 2,000 five-option questions (math 550, physics 510, chemistry 370, biology 370, informatics 200; grades 7 to 11). 413 held out for testing; of the rest, 1,408 for training, 169 for choosing the epoch, and 10 excluded because they were too similar to KazMMLU questions. The split was fixed by a hash of each id before any training.
- **Overlap check** (`overlap_check.ipynb`). Every question compared with all KazMMLU Kazakh STEM questions (test and dev) by wording (character 3–5-gram TF-IDF) and meaning (LaBSE). 10 flagged and excluded, 35 borderline kept; median nearest similarity 0.69.
- **Training** (`train_lora.ipynb`). LoRA r = 16, alpha = 16 on all attention and MLP projections; AdamW, learning rate 2e-4, 5% warm-up then linear decay; batch 8; 2 epochs, keeping the epoch with the best validation accuracy. The loss is cross-entropy on the answer letter only, using exactly the evaluation prompt. Base model in float16, adapter in float32, on 2 × T4 (Kaggle). Settings were fixed before training and identical for all seeds and both models.
- **Evaluation** (`eval_kazmmlu.ipynb`). Zero-shot with an English instruction, the question and options in Kazakh, and the model's chat template. The answer is the letter with the highest probability (the "A" and " A" token forms combined), among the options present. KazMMLU test split, five Kazakh STEM subjects and two Kazakh non-STEM subjects (Kazakh History, Geography).
- **Statistics**. Per-seed paired comparison with exact McNemar test and bootstrap interval (`compare.py`); three-seed mean with bootstrap intervals and Holm correction across subjects (`aggregate.py`); robustness checks (`robustness_checks.py`, `near_train_check.py`).

## Repository

All files are in the top folder.

| Files | What they are |
|---|---|
| `overlap_check.ipynb`, `train_lora.ipynb`, `eval_kazmmlu.ipynb` | Kaggle notebooks: similarity check against KazMMLU, LoRA training (one seed per run), scoring |
| `make_eval_nb.py`, `make_train_nb.py`, `make_overlap_nb.py`, `kazstem_common.py` | Scripts that generate the notebooks, with the shared prompt code |
| `validate_questions.py` | Checks used before release: answer keys, duplicates, balance, overlap |
| `hf_to_notebook_inputs.py` | Turns the Hugging Face dataset into the notebooks' input files |
| `results_<run>.csv` | One row per question per run: gold answer, prediction, letter probabilities |
| `compare_<model>_seed<N>.txt` | Per-seed paired comparison with the starting model |
| `three_seed_summary_<model>.csv/.txt` | Three-seed means, intervals, Holm-corrected subject p-values |
| `robustness_checks.*`, `near_train_check_<model>.*` | Two-level intervals, letter-habit correction, near-copy and look-alike checks |
| `overlap_top1.csv`, `overlap_flagged.csv`, `heldout_similarity.csv`, `training_runs.csv` | Overlap check results, held-out similarity, epoch kept per seed |
| `compare.py`, `aggregate.py`, `robustness_checks.py`, `near_train_check.py`, `plot_two_models.py` | Analysis scripts that produce these tables and the figure |

The result files list KazMMLU questions by their KazMMLU `ID` only; no KazMMLU question text is included.

## Reproduce

1. `python hf_to_notebook_inputs.py` writes `questions_clean.jsonl` and `test_own.csv`. Upload both to Kaggle as one dataset.
2. In Kaggle, import the three notebooks. Settings: GPU T4 x2, Internet on, the dataset as an input, and an `HF_TOKEN` secret with access to the gated Llama model (not needed for Qwen).
3. Starting scores: `eval_kazmmlu.ipynb` with `USE_ADAPTER = False`, `RUN_NAME = "base_llama"` or `"base_qwen"`.
4. Training: `train_lora.ipynb` with `MODEL = "llama"` or `"qwen"` and `SEED = 1`, `2`, `3`. Use Save Version → Save & Run All so the output is kept.
5. Scoring: `eval_kazmmlu.ipynb` with `USE_ADAPTER = True`, `RUN_NAME = "tuned_<model>_seed<N>"`, and that training run's output as the only adapter input. Score in a fresh session, not right after training.
6. Analysis, from the repository folder:

```bash
python compare.py results_base_qwen.csv results_tuned_qwen_seed1.csv
python aggregate.py --out=three_seed_summary_qwen.csv results_base_qwen.csv results_tuned_qwen_seed{1,2,3}.csv
python robustness_checks.py heldout_similarity.csv
python near_train_check.py --out=near_train_check_qwen.csv overlap_top1.csv results_base_qwen.csv results_tuned_qwen_seed{1,2,3}.csv
python plot_two_models.py
```

Versions used: torch 2.11.0, transformers 5.20.2, peft 0.21.2. Kaggle's preinstalled torchao 0.10 blocks peft; the notebooks uninstall it before loading peft.

## Limitations

- Two models of 7 to 8 billion parameters, one training recipe and one prompt format.
- The questions were partly written by me and partly drafted with an AI assistant; I checked every question and answer key myself. A shared style may be part of what the models learned.
- Answer keys were checked by one person.
- Only the Kazakh-language part of KazMMLU was used.

## Licenses

- Code: MIT (`LICENSE`).
- Questions: CC BY 4.0 (on Hugging Face).
- Llama adapter: Llama 3.1 Community License. Built with Llama.
- Qwen adapter: Apache 2.0.
- KazMMLU is CC BY-NC 4.0 and is not redistributed here.

## Citation

```bibtex
@misc{sanim09_kazstem_2026,
  author       = {Sanim09},
  title        = {Does fine-tuning on new Kazakh STEM questions improve KazMMLU?},
  year         = {2026},
  howpublished = {\url{https://github.com/sanim-science/kazakh-stem-llm}}
}
```

KazMMLU: Togmanov et al., *KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan*, 2025, [arXiv:2502.12829](https://arxiv.org/abs/2502.12829).

## Acknowledgements

Thanks to the KazMMLU authors at MBZUAI for releasing the benchmark, and to Kaggle for free GPU time. The notebooks and analysis code were developed with the help of Claude (Anthropic).
