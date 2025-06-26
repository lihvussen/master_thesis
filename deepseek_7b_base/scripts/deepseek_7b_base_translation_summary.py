import json
import statistics

# Set model name for path construction
model_all = "deepseek_7b_base"

# Load the translation evaluation scores JSON
with open(f"{model_all}/mexa/task_results/{model_all}_translation_evaluation.json", "r") as f:
    evaluations = json.load(f)

# List of languages to analyze grouped by language families (for clarity)
languages_to_analyze = [
    "Swahili", "Yoruba",      # African Languages
    "Hindi", "Urdu",          # South Asian Languages
    "German",                 # Germanic Languages
    "Italian", "Spanish",     # Romance Languages
    "Polish", "Croatian", "Russian",  # Slavic Languages
    "Hungarian",              # Other European Languages
]

# Initialize dictionaries to collect all scores per language
final_list_dict = {lang: [] for lang in languages_to_analyze}
final_dict = {lang: [] for lang in languages_to_analyze}

# Iterate over all examples and their language pair evaluations
for example, language_pairs in evaluations.items():
    for language_pair, eval_score in language_pairs.items():
        # Get source and target language codes
        source_lang, target_lang = language_pair.split("-")
        # Append the score to the source language list if in target list
        if source_lang in languages_to_analyze:
            final_list_dict[source_lang].append(eval_score[0])
        # Append the score to the target language list if in target list
        if target_lang in languages_to_analyze:
            final_list_dict[target_lang].append(eval_score[0])

# Calculate average evaluation score per language, rounded to 2 decimals
for lang, evals in final_list_dict.items():
    if evals:
        final_dict[lang] = round(statistics.mean(evals), 2)
    else:
        final_dict[lang] = None  # or 0 or skip, depending on preference

# Save the aggregated average scores back to a JSON file
with open(f"{model_all}/mexa/task_results/{model_all}_translation_results.json", "w", encoding="utf-8") as f:
    json.dump(final_dict, f, indent=4, ensure_ascii=False)
