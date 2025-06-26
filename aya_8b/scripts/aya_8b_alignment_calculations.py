import json
import statistics

# Define the types of alignment sources to evaluate
alignment_sources = ["cosine", "retrieval"]

# Define the different pooling or embedding extraction options
options = ["all", "last", "weighted"]

# Number of examples and iterations used in the evaluation
nr_examples = 100
iterations = 5

# Name of the model being evaluated
model = "aya_8b"

# Loop over each alignment source and each pooling option
for alignment_source in alignment_sources:
    for option in options:

        # Initialize a dictionary to store results for each iteration
        iteration_dict = {str(iter+1): {"mean_pooling": {}, "max_pooling": {}} for iter in range(iterations)}

        # Loop over each iteration to load and aggregate results
        for iteration in range(iterations):
            # Open the corresponding result file for this iteration
            with open(f"{model}/{alignment_source}_alignment/{option}_general_results/{model}_{alignment_source}_results_general_{option}_iteration_{iteration+1}_examples_{nr_examples}.json", "r") as f:
                iteration_result = json.load(f)

            # Extract all language pairs from the first layer (assumed to be "0")
            all_lang_pairs = list(iteration_result["0"].keys())

            # Initialize empty result lists for each language pair
            for one_lang_pair in all_lang_pairs:
                iteration_dict[str(iteration+1)]["mean_pooling"][one_lang_pair] = []
                iteration_dict[str(iteration+1)]["max_pooling"][one_lang_pair] = []

            # Loop through all layers and collect scores
            for layer, language_pairs in iteration_result.items():
                for language_pair, score in language_pairs.items():
                    # Try to add the score to the mean pooling list; if the key is missing, try the reversed language pair
                    try:
                        iteration_dict[str(iteration+1)]["mean_pooling"][language_pair].append(score)
                    except:
                        iteration_dict[str(iteration+1)]["mean_pooling"]["-".join([language_pair.split("-")[1], language_pair.split("-")[0]])].append(score)

                    # Same for max pooling
                    try:
                        iteration_dict[str(iteration+1)]["max_pooling"][language_pair].append(score)
                    except:
                        iteration_dict[str(iteration+1)]["max_pooling"]["-".join([language_pair.split("-")[1], language_pair.split("-")[0]])].append(score)

        # After gathering all scores, compute mean and max for each language pair
        for iter_nr, poolings in iteration_dict.items():
            for pooling, lang_pairs in poolings.items():
                for lang_pair, results in lang_pairs.items():
                    if pooling == "max_pooling":
                        iteration_dict[iter_nr][pooling][lang_pair] = round(max(results), 3)
                    if pooling == "mean_pooling":
                        iteration_dict[iter_nr][pooling][lang_pair] = round(statistics.mean(results), 3)

        # Write the aggregated results to a JSON file
        with open(f"{model}/{alignment_source}_alignment/alignment_results/{model}_alignment_results_{option}_{alignment_source}.json", "w") as f:
            json.dump(iteration_dict, f, indent=4, ensure_ascii=False)
