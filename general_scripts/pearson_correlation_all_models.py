import json
from scipy.stats import pearsonr  # For computing Pearson correlation coefficient

# List of models to analyze
models = ["aya_8b", "bloom_3b", "deepseek_7b_base", "gemma_7b", "llama2_7b", "llama3_1_8b", "mistral_7b_v01"]

# Different pooling/embedding extraction options for alignment
options = ["all", "last", "weighted"]

# Different evaluation tasks for which results exist
tasks = ["belebele", "mmlu", "translation"]

# Types of alignment extraction methods to compare against task performance
alignment_extractions = ["mexa", "cosine", "retrieval"]

# Languages to analyze specifically for MMLU task (mapping full name to language code)
languages_to_analyze_dict_mmlu = {
    "Croatian": "hr", "Russian": "ru",  # Slavic Languages
    "Hindi": "hi",
    "English": "en", "German": "de",    # Germanic Languages
    "Italian": "it", "Spanish": "es",   # Romance Languages
    "Hungarian": "hu",                  # Other European Languages
}

# Loop over each alignment extraction method
for alignment_extraction in alignment_extractions:
    # Loop over each evaluation task
    for task in tasks:
        models_dict = {}  # To store correlations for all models

        # Loop over each model
        for model in models:
            # Load task performance results (per language) for the current model and task
            with open(f"{model}/mexa/task_results/{model}_{task}_results.json", "r") as f:
                task_results = json.load(f)

            option_dict = {}  # Store correlations by pooling option for this model

            # Loop over each pooling option ('all', 'last', 'weighted')
            for option in options:
                iteration_dict = {}  # Store correlations by iteration for this option

                # For file path consistency: if alignment_extraction is 'mexa', use directly,
                # otherwise append '_alignment' suffix to folder name
                if alignment_extraction == "mexa":
                    al_ext = alignment_extraction
                else:
                    al_ext = f"{alignment_extraction}_alignment"

                # Load alignment results for the current model, option, and alignment extraction method
                with open(f"{model}/{al_ext}/alignment_results/{model}_alignment_results_{option}_{alignment_extraction}.json", "r") as f:
                    alignment_results = json.load(f)

                # Loop over iterations in the alignment results
                for iteration, poolings in alignment_results.items():
                    pooling_dict = {}  # Store correlation per pooling type (e.g. 'all', 'last')

                    # Loop over each pooling method in the results (should match options)
                    for pooling, language_pairs in poolings.items():
                        new_dict = {}  # Store filtered language pair data: task performance + alignment score

                        # Filter language pairs containing "English" to correlate non-English task scores with alignment
                        for language_pair, alignment in language_pairs.items():
                            if "English" in language_pair:
                                # Handle two possible separators in language pairs: underscore or dash
                                if "_" in language_pair:
                                    languages = language_pair.split("_")
                                elif "-" in language_pair:
                                    languages = language_pair.split("-")
                                else:
                                    languages = [language_pair]

                                for language in languages:
                                    # Skip English itself; focus on other languages in the pair
                                    if language == "English":
                                        continue

                                    # For the "mmlu" task, only include languages in the predefined dictionary to analyze
                                    if task == "mmlu":
                                        if language in languages_to_analyze_dict_mmlu.keys():
                                            # Save a list: [task performance, alignment score] for this language pair
                                            new_dict[language_pair] = [task_results[language], alignment]
                                    else:
                                        # For other tasks, include all non-English languages without filtering
                                        new_dict[language_pair] = [task_results[language], alignment]

                        # Extract separate lists of task performances and alignment scores
                        first_values = [v[0] for v in new_dict.values()]  # task results
                        second_values = [v[1] for v in new_dict.values()]  # alignment scores

                        try:
                            # Compute Pearson correlation between task performance and alignment scores
                            correlation, _ = pearsonr(first_values, second_values)
                        except Exception as e:
                            # Print values if correlation computation fails for debugging
                            print(first_values)
                            print(second_values)

                        # Store rounded correlation coefficient for this pooling method
                        pooling_dict[pooling] = round(correlation, 2)

                    # Store correlations by pooling for this iteration
                    iteration_dict[iteration] = pooling_dict

                # Store all iteration data for this pooling option
                option_dict[option] = iteration_dict

            # Store all options data for this model
            models_dict[model] = option_dict

        # Save all computed correlations (all models, options, iterations) as JSON for the current alignment extraction and task
        with open(f"general_results/{alignment_extraction}/{task}/pearson_correlation_all_models_{alignment_extraction}.json", "w") as f:
            json.dump(models_dict, f, indent=4)

        print("Done")
