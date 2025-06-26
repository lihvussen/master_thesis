import json
import statistics

# Experiment parameters
number_of_iterations = 5
number_of_examples = 100
model = "bloom_3b"  # Consistent model name reference

options = ["all", "last", "weighted"]

for option in options:
    option_results = {}  # Store results for all iterations of current option

    for iteration in range(number_of_iterations):
        # Initialize structure to hold pooled alignment scores
        iteration_results = {"mean_pooling": {}, "max_pooling": {}}

        # Construct input file path
        input_path = f"{model}/mexa/{option}_general_results/{model}_mexa_results_general_{option}_iteration_{iteration+1}_examples_{number_of_examples}.json"

        # Load current iteration data
        with open(input_path, "r") as f:
            iteration_data = json.load(f)
            language_pairs = list(iteration_data["0"].keys())  # Extract from first layer

        # Initialize per-language-pair score collector
        alignment_scores = {lp: [] for lp in language_pairs}

        for layer_data in iteration_data.values():
            for lang_pair, score in layer_data.items():
                try:
                    alignment_scores[lang_pair].append(score)
                except KeyError:
                    # Handle flipped language pair case
                    lang1, lang2 = lang_pair.split("_")
                    flipped_pair = f"{lang2}_{lang1}"
                    alignment_scores[flipped_pair].append(score)

        # Compute mean and max pooling for each pair
        for pair, scores in alignment_scores.items():
            iteration_results["mean_pooling"][pair] = round(statistics.mean(scores), 2)
            iteration_results["max_pooling"][pair] = round(max(scores), 2)

        option_results[str(iteration + 1)] = iteration_results

    # Output results to JSON
    output_path = f"{model}/mexa/alignment_results/{model}_alignment_results_{option}_mexa.json"
    with open(output_path, "w") as f:
        json.dump(option_results, f, indent=4)
