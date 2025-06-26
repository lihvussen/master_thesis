This repository contains the pipeline and the structure that was used for carrying out the master thesis experiment aimed at measuring multilingual alignment in LLMS. In order to complete reproduce the pipeline the following steps have to be completed.

First, the "flores_retriever.py" script downloads the FLORES-101 dataset with the use of flores_101_languages.json file which containes abbreviations of languages and the respective languages.

Then, there seven folders named with the respective model names that were analyzed in the project - aya_8b, bloom_3b, deepseek_7b_base, gemma_7b, llama2_7b, llama3_1_8b and mistral_7b_v01. Each of these folders has the same structure, so the general logic of each of them them will be presented.

In each of these folders there is a "scripts" folder. It contains all the scripts used for this particular model. The first one which should be executed is the [model_name]_representaion_retrieval.py script. It retrieves the embeddings for all the selected languages and sequences and saves them into the following folders: hidden_state_averages (for non-weighted average embedding), hidden_state_last (for last token embedding) and hidden_state_weighted (for weighted average embedding).

Then the [model_name]_cosine_similarities.py calculates pairwise cosine similarity between hidden state representations (vectors) from different languages for a particular model, across multiple pooling strategies, layers, and iterations and stores them in the cosine_similarities folder in the respective folders depending on the representation extraction strategy. After that, The [model_name]_cosine_to_alignment.py collects and saves all raw cosine similarity values per language pair and per layer (without averaging), allowing for detailed statistical analysis and accounting for reversed language pairs. This results are saved in language_pairs_layers_similarities folder, in the in the respective folders depending on the representation extraction strategy. This data is used then to create plots with the [model_name]_plot_creation.py script, which show how the alignment particular language pairs behaves among the layers. The plots are in the plots folder. 

In contrast, the [model_name]_cosine_analysis.py computes and stores the mean cosine similarity per language pair and layer, summarizing the data into a compact form. These results, also depending on the representation extraction strategy are then stored in the cosine_alignment folder. The files produced by this script will be used for measuring the correlation between the alignment and performance on downstream tasks.

Parallelly, the [model_name]_similarities_for_sentence_retrieval.py scirpts computes sentence retrieval scores across 5 iterations and 3 embedding strategies. For each layer, it loads hidden state vectors, computes cosine similarity between all language-example combinations, and retrieves the top most similar examples (excluding self-similarity). Results are saved per layer and iteration, enabling detailed analysis of cross-lingual retrieval quality by layer and embedding strategy. The files are saved in the retrieval folder. Then, the [model_name]_retrieval_statistics.py evaluates how well a model retrieves representations from the same language and the same translated sentence (semantic match) across layers, iterations, and embedding options. It computes the percentage of correct matches in the top retrieved items and their average rank positions to assess multilingual alignment and retrieval quality. The data is saved in percentages_of_most_similar_representations_same_example_and_language folder and is then used by [model_name]_plot_creation_sentence_retrieval.py script to create plots that aggreagate the overall alignment of each representation extraction method and save them in the plots_retrieval folder.

The data saved in the retrieval folder are also used in the [model_name]_retrieval_alignment_calculation.py, which saves the alignment scores in the retrieval_alignment folder similarly to the cosine_alignment folder. This data will also be used for measuring the correlation between the alignment and performance on downstream tasks.

At the same time, the [model_name]_retrieval_alignment_calculation_for_plots.py should be used to create the data for the alignment plots parallel to the ones for cosine similarity. This data is saved in the language_pairs_layers_retrieval folder and the [model_name]_plot_creation_sentence_retrieval.py creates these plots and saves them in the plots_retrieval_language_pair_alignment.py folder.

Also paralelly to that the MEXA pipeline should be executed. First, the [model_name]_mexa_reshuffle_calculations.py reshuffles the hidden states data first and then calculates MEXA scores later. They are saved in the MEXA folder and then in the appropriate subfolder depending on the representation extraction method. Then, this data is used by [model_name]_alignment_results.py in order to compute the final aignment scores and save them in the alignment_scores subfolder of the mexa folder. At the same time the [model_name]_alignment_calculations.py folder should be used to calculate the same thing for the cosine similarity and sentence retrieval measurements. Their scores are also saved in the the alignment_results subfolder of their [measurement]_alignment folders. Then, the [model_name]_mexa_alignment_calculation_for_plots.py prepares the data for plotting and saves them in the language_pairs_layers_mexa folder and these data are plotted with the [model_name]_plot_creation_mexa.py and saved in the plots_mexa folder.

The tasks should be then calculated with [model_name]_belebele.py, [model_name]_mmlu.py and [model_name]_translation.py scripts the results are saved in the task_results subfolder of the mexa folder. For translation task, the [model_name]_translation_evaluation.py uses COMET to evaluate these translations and [model_name]_translation_summary.py creates the final scores of these translations. Finally the [model_name]_plot_creation_tasks.py plots the results for each file and also saves them in the task_results subfolder.

Finally, the [model_name]_alignment_top.py script should be executed which retrieves n best and worst language pairs for each model and each measurement and saves them in the alignment_results subfolder of each measurement task. This file was used for the table creation of the best and worst performing language pairs for each model.

The second part of scripts is contained by the general_scripts folder, which contains scripts where all the models are analyzed. First of all, the pearson_correlation_all_models.py calculates the correlation between the alignment and downstream task performance and saves them in the respective cosine, mexa and retrieval folders and their task subfolders. This data is then plotted with pearson_correlation_plotting.py and all the plots are saved in the same, respective subfolders as the data.

Other scripts in the general_scripts folder were used as helper functions. The plots_averaged_task_results.py creates the plot_all_languages_merged.png in the general_results folder with all the task results for each model aggreated. The plots_merging.py merged the alignment plots from different representation extraction strategy for each model in the plots_merged_rows subfolder of the appropriate measurement method folder in general_results. These merged plots were presented in the appendix. Finally, the aggregating_alignment_plots.py summed up alignment of each model for each measurment and aggreagated them into one plot in the alignment_aggregated subfolder of the general_results folder.

Remarks:
Many folders contain three subfolders with the following names: all, last and weighted. They refer to the representation extraction method. The 'all', also in the code, referes to non-weighted averaged embedding.

The repo contains all the plots and some json files with the results. Most of them were not upload due to their extensive size.

---

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
