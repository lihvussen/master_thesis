import json
from itertools import combinations
import time

# Define experiment parameters
nr_examples = 100
iterations = 5
model = "aya_8b"

start = time.time()

# Languages selected for MEXA evaluation
languages_to_analyze = [
    "Swahili", "Yoruba",  # African Languages
    "Hindi", "Urdu",
    "German", "English",  # Germanic Languages
    "Italian", "Spanish",  # Romance Languages
    "Polish", "Croatian", "Russian",  # Slavic Languages
    "Hungarian",  # Other European Languages
]

# Generate all unique language pairs
language_pairs = ["-".join(pair) for pair in combinations(languages_to_analyze, 2)]

# Embedding strategies to analyze
options = ["all", "last", "weighted"]

for option in options:
    for iteration in range(iterations):

        # Load per-example MEXA results for the given option and iteration
        input_path = f"{model}/mexa/{option}_general_results/{model}_mexa_results_general_{option}_iteration_{iteration+1}_examples_{nr_examples}.json"
        with open(input_path, "r") as f:
            layers = json.load(f)

        # Initialize dictionary to store scores per language pair per layer
        pairs_layers_dict = {lp: {} for lp in language_pairs}

        # Iterate through layers and example language pairs to compute relative ranking scores
        for layer, example_languages in layers.items():
            for example_language, example_language_results in example_languages.items():
                for i, (example_language_result, _) in enumerate(example_language_results.items()):
                    # Only compare translation pairs (same sentence index)
                    if example_language.split('_')[1] == example_language_result.split('_')[1]:
                        ex_lang = example_language.split('_')[0]
                        ex_lang_res = example_language_result.split('_')[0]

                        # Normalize pair direction (A-B or B-A)
                        pair = f"{ex_lang}-{ex_lang_res}" if f"{ex_lang}-{ex_lang_res}" in language_pairs else f"{ex_lang_res}-{ex_lang}"
                        score = round(1 - (i + 1) / 1200, 4)

                        # Append score for the current layer and language pair
                        if layer not in pairs_layers_dict[pair]:
                            pairs_layers_dict[pair][layer] = []
                        pairs_layers_dict[pair][layer].append(score)

        # Save computed scores to file
        output_path = f"{model}/language_pairs_layers_mexa/{option}/{model}_language_pairs_layers_mexa_{iteration+1}_iteration_{nr_examples}_examples_{option}.json"
        with open(output_path, "w") as f:
            json.dump(pairs_layers_dict, f, indent=4, ensure_ascii=False)

        print(f"Iteration {iteration+1} for option {option} finished.")

    end = time.time()
    print(f"Option {option} finished in {round((end - start) / 60, 2)} minutes.")
