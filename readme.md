# Multilingual Alignment Evaluation in LLMs

This repository contains the pipeline and folder structure used in a master thesis experiment focused on measuring **multilingual alignment in large language models (LLMs).**

---

## 📦 Repository Structure

### 📁 Model Folders

The repository includes seven folders named after the models analyzed:

- `aya_8b`
- `bloom_3b`
- `deepseek_7b_base`
- `gemma_7b`
- `llama2_7b`
- `llama3_1_8b`
- `mistral_7b_v01`

Each folder contains:
- `scripts/`: Scripts specific to the model
- Subfolders for storing results (e.g., `hidden_state_averages`, `cosine_alignment`, `plots`, etc.)

---

## 🧪 Pipeline Overview

### Step 1: Download FLORES-101 Dataset

Run:

```bash
python flores_retriever.py
This script uses flores_101_languages.json to download the dataset with the correct language abbreviations.

Step 2: Embedding Retrieval
Run:

bash
Kopiuj
Edytuj
python [model_name]_representation_retrieval.py
Saves embeddings to:

hidden_state_averages/ (non-weighted average)

hidden_state_last/ (last token)

hidden_state_weighted/ (weighted average)

Step 3: Cosine Similarity Analysis
Pairwise Cosine Similarity

bash
Kopiuj
Edytuj
python [model_name]_cosine_similarities.py
➜ Output: cosine_similarities/

Save Raw Similarities

bash
Kopiuj
Edytuj
python [model_name]_cosine_to_alignment.py
➜ Output: language_pairs_layers_similarities/

Plot Layer-wise Alignment

bash
Kopiuj
Edytuj
python [model_name]_plot_creation.py
➜ Output: plots/

Summarized Cosine Alignment

bash
Kopiuj
Edytuj
python [model_name]_cosine_analysis.py
➜ Output: cosine_alignment/

Step 4: Sentence Retrieval Evaluation
Compute Retrieval Scores

bash
Kopiuj
Edytuj
python [model_name]_similarities_for_sentence_retrieval.py
➜ Output: retrieval/

Statistics on Retrieval

bash
Kopiuj
Edytuj
python [model_name]_retrieval_statistics.py
➜ Output: percentages_of_most_similar_representations_same_example_and_language/

Plot Sentence Retrieval Alignment

bash
Kopiuj
Edytuj
python [model_name]_plot_creation_sentence_retrieval.py
➜ Output: plots_retrieval/

Save Retrieval Alignment Scores

bash
Kopiuj
Edytuj
python [model_name]_retrieval_alignment_calculation.py
➜ Output: retrieval_alignment/

Plot Retrieval Alignment by Language Pairs

bash
Kopiuj
Edytuj
python [model_name]_retrieval_alignment_calculation_for_plots.py
python [model_name]_plot_creation_sentence_retrieval.py
➜ Outputs:

language_pairs_layers_retrieval/

plots_retrieval_language_pair_alignment/

Step 5: MEXA Score Calculation
MEXA Computation

bash
Kopiuj
Edytuj
python [model_name]_mexa_reshuffle_calculations.py
➜ Output: MEXA/[strategy]/

Final Alignment Scores

bash
Kopiuj
Edytuj
python [model_name]_alignment_results.py
python [model_name]_alignment_calculations.py
➜ Outputs:

MEXA/alignment_scores/

[cosine|retrieval]_alignment/alignment_results/

Plot MEXA Results

bash
Kopiuj
Edytuj
python [model_name]_mexa_alignment_calculation_for_plots.py
python [model_name]_plot_creation_mexa.py
➜ Outputs:

language_pairs_layers_mexa/

plots_mexa/

Step 6: Downstream Task Evaluation
Run:

bash
Kopiuj
Edytuj
python [model_name]_belebele.py
python [model_name]_mmlu.py
python [model_name]_translation.py
python [model_name]_translation_evaluation.py
python [model_name]_translation_summary.py
python [model_name]_plot_creation_tasks.py
➜ Output: MEXA/task_results/

Step 7: Alignment Top-K Analysis
Run:

bash
Kopiuj
Edytuj
python [model_name]_alignment_top.py
➜ Output: Top-N best/worst aligned language pairs per model saved in:

alignment_results/ subfolders of each measurement.

🔄 General Scripts (Across Models)
Located in the general_scripts/ folder:

Correlation with Tasks

bash
Kopiuj
Edytuj
python pearson_correlation_all_models.py
python pearson_correlation_plotting.py
➜ Output: Correlation data + plots in each measurement method’s folder

Helper Scripts:

plots_averaged_task_results.py ➜ Generates plot_all_languages_merged.png in general_results/

plots_merging.py ➜ Merges plots for each extraction method

aggregating_alignment_plots.py ➜ Summarizes model alignment per metric

📁 Notes on Folder Naming
Many result folders contain three subfolders:

all/ → non-weighted average embedding

last/ → last token embedding

weighted/ → weighted average embedding

In code, "all" corresponds to "non-weighted average."

📂 Data & Results
The repo contains all scripts, folder structure, and example plots. Some .json and .pt results files were not uploaded due to size constraints, but structure and names remain intact.

📌 Summary
This repo allows full reproduction of the thesis experiments and provides:

Per-model multilingual alignment scores

Cross-layer analyses

Embedding strategy comparisons

Correlations with real-world downstream performance

📍 To reproduce everything, follow each section above for every model
