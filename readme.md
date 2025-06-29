Here is the revised and polished `README.md` with `[model_name]_` prepended to each script name where applicable. This helps clarify that each model has its own set of scripts prefixed with its name:

---

# 🌐 Multilingual Alignment in LLMs – Thesis Experiment Pipeline

This repository contains the complete pipeline used in a master's thesis evaluating **multilingual alignment in Large Language Models (LLMs)**. It supports reproducible analysis across several models, embedding strategies, and alignment metrics.

---

## 📦 Repository Overview

The pipeline evaluates these multilingual models:

* `aya_8b`
* `bloom_3b`
* `deepseek_7b_base`
* `gemma_7b`
* `llama2_7b`
* `llama3_1_8b`
* `mistral_7b_v01`

Each model has its own folder with a `scripts/` directory containing all related scripts. All models follow the same workflow and folder structure.

---

## 📥 Dataset Preparation

Run:

* `flores_retriever.py`

Downloads the **FLORES-101** dataset.
The file `flores_101_languages.json` provides language names and codes used in all experiments.

---

## 📁 Folder Structure

Each model directory contains the following:

| Folder                                                                   | Description                                                |
| ------------------------------------------------------------------------ | ---------------------------------------------------------- |
| `hidden_states_averages/`                                                | Embeddings using non-weighted average pooling              |
| `hidden_states_last/`                                                    | Embeddings from the last token                             |
| `hidden_states_weighted/`                                                | Attention-weighted average embeddings                      |
| `cosine_similarities/`                                                   | Raw cosine similarity values for all strategies and layers |
| `language_pairs_layers_similarities/`                                    | Cosine similarities between all language pairs and layers  |
| `cosine_alignment/`                                                      | Averaged cosine alignment scores                           |
| `retrieval/`                                                             | Sentence retrieval results                                 |
| `retrieval_alignment/`                                                   | Averaged retrieval alignment values                        |
| `percentages_of_most_similar_representations_same_example_and_language/` | Retrieval accuracy (same example/language)                 |
| `plots/`                                                                 | Cosine alignment plots                                     |
| `plots_retrieval/`                                                       | Sentence retrieval summary plots                           |
| `plots_retrieval_language_pair_alignment/`                               | Retrieval alignment plots per language pair                |
| `language_pairs_layers_retrieval/`                                       | Layer-wise retrieval scores per language pair              |
| `mexa/`                                                                  | Shuffled embeddings and intermediate MEXA results          |
| `language_pairs_layers_mexa/`                                            | Layer-wise MEXA scores per language pair                   |
| `plots_mexa/`                                                            | MEXA alignment plots                                       |
| `scripts/`                                                               | All scripts for running this model's pipeline              |

---

## 🧠 Embedding Extraction

Run:

* `[model_name]_representation_retrieval.py`

Generates contextual embeddings (average, last token, attention-weighted) and saves them in their respective `hidden_states_*` folders.

---

## 🔁 Cosine Similarity Alignment

Run:

* `[model_name]_cosine_similarities.py`
* `[model_name]_cosine_to_alignment.py`
* `[model_name]_cosine_analysis.py`
* `[model_name]_plot_creation.py`

These scripts compute and visualize alignment based on cosine similarity between translated sentence embeddings.

---

## 🔍 Sentence Retrieval Alignment

Run:

* `[model_name]_similarities_for_sentence_retrieval.py`
* `[model_name]_retrieval_statistics.py`
* `[model_name]_retrieval_alignment_calculation.py`
* `[model_name]_retrieval_alignment_calculation_for_plots.py`
* `[model_name]_plot_creation_sentence_retrieval.py`

These evaluate how well the model retrieves correct translations from a pool of examples using cosine similarity.

---

## 🧪 MEXA Alignment

Run:

* `[model_name]_mexa_reshuffle_calculations.py`
* `[model_name]_alignment_results.py`
* `[model_name]_alignment_calculations.py`
* `[model_name]_mexa_alignment_calculation_for_plots.py`
* `[model_name]_plot_creation_mexa.py`

These scripts compute the MEXA (Multilingual Example Alignment) scores using shuffled embeddings as a baseline.

---

## 📉 Downstream Task Evaluation

Run:

* `[model_name]_belebele.py`
* `[model_name]_mmlu.py`
* `[model_name]_translation.py`
* `[model_name]_translation_evaluation.py`
* `[model_name]_translation_summary.py`
* `[model_name]_plot_creation_tasks.py`

These evaluate downstream performance using:

* **Belebele** (QA)
* **MMLU** (reasoning)
* **Translation** (COMET scores)

---

## 🏅 Language Pair Alignment Ranking

Run:

* `[model_name]_alignment_top.py`

Finds top-N best and worst aligned language pairs for each method (cosine, retrieval, MEXA).

---

## 🔬 Cross-Model Analysis (`general_scripts/`)

### 📈 Correlation with Task Performance

Run:

* `pearson_correlation_all_models.py`
* `pearson_correlation_plotting.py`

Computes correlation between alignment metrics and downstream task scores across all models.

---

### 📊 Aggregate Plotting Tools

Run:

* `plots_averaged_task_results.py`
* `plots_merging.py`
* `aggregating_alignment_plots.py`

Generates merged plots across models and alignment strategies.

---

## 📁 Representation Strategies

Folders are divided by pooling strategy:

| Subfolder   | Description                |
| ----------- | -------------------------- |
| `all/`      | Non-weighted average       |
| `last/`     | Last-token embedding       |
| `weighted/` | Attention-weighted average |

**Note**: `"all"` corresponds to the default average strategy used in most plots and evaluations.

---

## ✅ Summary

This pipeline evaluates multilingual alignment using:

* Cosine similarity
* Sentence-level retrieval
* MEXA reshuffling-based alignment
* Correlation with downstream tasks (Belebele, MMLU, Translation)

All experiments are performed layer-wise and across different embedding strategies.

---

Let me know if you'd like a downloadable `.md` version or if you want to include emojis, badges, or automatic TOC at the top!
