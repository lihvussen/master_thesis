import json
from itertools import combinations
import time

# Number of examples and iterations to process
nr_examples = 100
iterations = 5

start = time.time()

# Define model name for reuse in file paths
model = "mistral_7b_v01"

# List of languages to analyze (grouped loosely by linguistic families)
languages_to_analyze = [
    "Swahili", "Yoruba",  # African Languages
    "Hindi", "Urdu",
    "German", "English",  # Germanic Languages
    "Italian", "Spanish",  # Romance Languages
    "Polish", "Croatian", "Russian",  # Slavic Languages
    "Hungarian",  # Other European Languages
]

# Generate all unique language pairs as strings joined by '-'
language_pairs = ["-".join(pair) for pair in combinations(languages_to_analyze, 2)]

# Different embedding pooling strategies to process
options = ["all", "last", "weighted"]

# Loop over each embedding option
for option in options:

    # Loop over each iteration
    for iteration in range(iterations):

        # Load the retrieval cosine similarity results for this option and iteration
        with open(
            f"{model}/retrieval/{option}/"
            f"cosine_similarities_retrieval_{model}_{iteration+1}_iteration_{nr_examples}_examples_{option}.json", "r"
        ) as f:
            layers = json.load(f)

        # Initialize a dict to store scores per language pair and layer
        pairs_layers_dict = {lp: {} for lp in language_pairs}

        # Iterate over layers in the loaded JSON
        for layer, example_languages in layers.items():
            # For each language of the example in this layer
            for example_language, example_language_results in example_languages.items():
                # For each retrieved example and its similarity rank
                for i, (example_language_result, similarity) in enumerate(example_language_results.items()):
                    # Check if the target language code matches (split by '_' to separate language and something else)
                    if example_language.split('_')[1] == example_language_result.split('_')[1]:
                        # Extract language names before the '_'
                        ex_lang = example_language.split('_')[0]
                        ex_lang_res = example_language_result.split('_')[0]

                        # Create a pair string in consistent order (lex order fallback)
                        if f"{ex_lang}-{ex_lang_res}" in language_pairs:
                            pair = f"{ex_lang}-{ex_lang_res}"
                        else:
                            pair = f"{ex_lang_res}-{ex_lang}"

                        # Compute a score based on rank (1-based)
                        # The score formula: 1 - rank / 1200, rounded to 4 decimals
                        score = round(1 - (i + 1) / 1200, 4)

                        # Initialize list if layer not yet present for this pair
                        if layer not in pairs_layers_dict[pair]:
                            pairs_layers_dict[pair][layer] = []

                        # Append the score to this language pair's layer list
                        pairs_layers_dict[pair][layer].append(score)

        # Define output file path and save the aggregated retrieval scores
        output_path = (
            f"{model}/language_pairs_layers_retrieval/{option}/"
            f"{model}_language_pairs_layers_retrieval_{iteration+1}_iteration_100_examples_{option}.json"
        )
        with open(output_path, "w") as f:
            json.dump(pairs_layers_dict, f, indent=4, ensure_ascii=False)

        print(f"Iteration {iteration+1} for option {option} finished.")

    end = time.time()
    print(f"Option {option} finished in {round((end - start) / 60, 2)} minutes.")
