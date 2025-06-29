# Multilingual Alignment in LLMs – Thesis Experiment Pipeline

This repository contains the pipeline and structure used for conducting the master’s thesis experiment aimed at measuring **multilingual alignment in Large Language Models (LLMs)**. To fully reproduce the pipeline, the following steps must be followed.

---

## 📥 Dataset Preparation

Run `flores_retriever.py` to download the **FLORES-101** dataset using the `flores_101_languages.json` file, which contains language abbreviations and their full names.

---

## 📁 Model Structure

Each of the following model folders is structured identically:

* `aya_8b`
* `bloom_3b`
* `deepseek_7b_base`
* `gemma_7b`
* `llama2_7b`
* `llama3_1_8b`
* `mistral_7b_v01`

Each folder contains a `scripts/` directory with the following workflow:

---

## 🧠 Embedding Extraction

Run:

```
[model_name]_representation_retrieval.py
```

This script extracts hidden state embeddings for each language and stores them in:

* `hidden_state_averages/` – non-weighted average embeddings (`all`)
* `hidden_state_last/` – last-token embeddings
* `hidden_state_weighted/` – attention-weighted average embeddings

---

## 🔁 Cosine Similarity Analysis

Run the following:

```
[model_name]_cosine_similarities.py
[model_name]_cosine_to_alignment.py
[model_name]_cosine_analysis.py
```

This computes pairwise cosine similarities between representations from different languages across layers and pooling strategies. Results are saved in:

* `cosine_similarities/`
* `language_pairs_layers_similarities/`
* `cosine_alignment/`

Then visualize the results with:

```
[model_name]_plot_creation.py
```

Plots are saved in `plots/`.

---

## 🔍 Sentence Retrieval Alignment

Run the following:

```
[model_name]_similarities_for_sentence_retrieval.py
[model_name]_retrieval_statistics.py
[model_name]_plot_creation_sentence_retrieval.py
[model_name]_retrieval_alignment_calculation.py
[model_name]_retrieval_alignment_calculation_for_plots.py
```

This computes sentence-level retrieval across languages and calculates multilingual alignment via retrieval. Outputs include:

* `retrieval/`
* `percentages_of_most_similar_representations_same_example_and_language/`
* `retrieval_alignment/`
* `language_pairs_layers_retrieval/`
* `plots_retrieval/`
* `plots_retrieval_language_pair_alignment/`

---

## 📊 MEXA Alignment

Run the following:

```
[model_name]_mexa_reshuffle_calculations.py
[model_name]_alignment_results.py
[model_name]_alignment_calculations.py
[model_name]_mexa_alignment_calculation_for_plots.py
[model_name]_plot_creation_mexa.py
```

This produces MEXA alignment scores based on reshuffled hidden states. Outputs:

* `MEXA/`
* `alignment_scores/` (within `MEXA/`)
* `language_pairs_layers_mexa/`
* `plots_mexa/`

---

## 🧪 Downstream Task Evaluation

Run:

```
[model_name]_belebele.py
[model_name]_mmlu.py
[model_name]_translation.py
[model_name]_translation_evaluation.py
[model_name]_translation_summary.py
[model_name]_plot_creation_tasks.py
```

Evaluates each model on:

* **Belebele** (QA)
* **MMLU** (Reasoning)
* **Translation** (evaluated using COMET)

Results saved in: `task_results/`

---

## 🏅 Top-N Alignment Results

Run:

```
[model_name]_alignment_top.py
```

Extracts top-N best and worst aligned language pairs for each model and metric. Output saved in `alignment_results/` of each method folder.

---

## 🔬 Cross-Model Analysis (`general_scripts/`)

### Correlation with Downstream Tasks

Run:

```
pearson_correlation_all_models.py
pearson_correlation_plotting.py
```

This computes Pearson correlation between alignment and downstream task performance. Results and plots are stored in:

* `general_results/cosine/`
* `general_results/retrieval/`
* `general_results/mexa/`

### Aggregated Plotting Tools

* `plots_averaged_task_results.py`: aggregates task scores across models into `plot_all_languages_merged.png`
* `plots_merging.py`: merges alignment plots across strategies into `plots_merged_rows/`
* `aggregating_alignment_plots.py`: generates an overview of alignment scores per model and metric

---

## 📁 Folder Naming Notes

Many folders contain subfolders:

* `all/` → non-weighted average embeddings
* `last/` → last-token embeddings
* `weighted/` → attention-weighted embeddings

> Note: In code, `"all"` refers to non-weighted average embeddings.

---

## ⚠️ Repository Contents

The repository contains:

* All final plots
* Some `.json` result files
* Large raw files and outputs are not included due to size constraints

---

## ✅ Summary

This repository supports a full reproduction of the following multilingual alignment evaluation pipeline:

* Cosine similarity
* Sentence retrieval
* MEXA reshuffling
* Correlation with downstream tasks (Belebele, MMLU, Translation)
* Multi-layer and multi-representation evaluation
* Per-model and cross-model visualization

---

## 📬 Contact

For questions or collaboration inquiries, feel free to open an issue or reach out to the maintainer.

---

Let me know if you want this split across multiple markdown files, with a live GitHub-style ToC, or further automated integration.
