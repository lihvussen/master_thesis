import json
import os

# Set the model name as a variable for easy reuse and modification
model = "llama2_7b"

number_of_iterations = 5
number_of_examples = 100
number_of_languages = 12

options = ["all", "last", "weighted"]

for option in options:

    for iteration in range(number_of_iterations):

        # Construct the output file path for this model, option, and iteration
        output_path = (
            f"{model}/percentages_of_most_similar_representations_same_example_and_language/"
            f"{option}/{model}_percentages_most_similar_retrieved_representations_"
            f"{iteration+1}_iteration_{number_of_examples}_examples_top_100_{option}.json"
        )

        # Skip computation if results for this iteration and option already exist
        if os.path.exists(output_path):
            print(f"Iteration {iteration+1} for option {option} already done")
            continue

        # Load cosine similarities for the current iteration and option
        with open(
            f"{model}/retrieval/{option}/cosine_similarities_retrieval_"
            f"{model}_{iteration+1}_iteration_{number_of_examples}_examples_{option}.json",
            "r"
        ) as f:
            current_iteration = json.load(f)

        all_layers = {}

        # Iterate over each layer in the cosine similarity data
        for layer, languages_examples in current_iteration.items():

            layer_dict = {}

            # Initialize counters and accumulators for stats:
            # language_stats[0] = count of same-language matches in top results
            # language_stats[1] = total possible same-language matches
            # example_stats[0] = count of same-text matches in top results
            # example_stats[1] = total possible same-text matches
            language_stats = [0, 0]
            language_positions = [0, 0]
            example_stats = [0, 0]
            example_positions = [0, 0]

            # Iterate over each query language-example pair and its retrieved results
            for language_example, results in languages_examples.items():
                for i, (language_example_2, similarity) in enumerate(results.items()):
                    # Check top number_of_examples-1 results for same language match
                    if i < number_of_examples - 1:
                        if language_example.split("_")[0] == language_example_2.split("_")[0]:
                            language_stats[0] += 1

                    # Check top number_of_languages-1 results for same text match
                    if i < number_of_languages - 1:
                        if language_example.split("_")[1] == language_example_2.split("_")[1]:
                            example_stats[0] += 1

                    # Accumulate positions (rank + 1) of matches for averaging later
                    if language_example.split("_")[0] == language_example_2.split("_")[0]:
                        language_positions[0] += i + 1
                    if language_example.split("_")[1] == language_example_2.split("_")[1]:
                        example_positions[0] += i + 1

                # Total counts for normalization
                language_stats[1] += number_of_examples - 1
                example_stats[1] += number_of_languages - 1
                language_positions[1] += number_of_examples - 1
                example_positions[1] += number_of_languages - 1

            # Print summary statistics for this layer
            print(
                f"For layer {layer} retrieved {language_stats[0]} out of {language_stats[1]} examples "
                f"for languages and {example_stats[0]} out of {example_stats[1]} examples for same texts."
            )

            # Calculate and store percentages and average ranks for this layer
            layer_dict[f"Percentage of records from same language in top {number_of_examples-1} records"] = round(language_stats[0] / language_stats[1], 3)
            layer_dict[f"Percentage of records from same text in top {number_of_languages-1} records"] = round(example_stats[0] / example_stats[1], 3)
            layer_dict[f"Average postion of a retrieved text in a language"] = round(language_positions[0] / language_positions[1], 1)
            layer_dict[f"Average postion of a retrieved text with the same semantic"] = round(example_positions[0] / example_positions[1], 1)

            all_layers[layer] = layer_dict

        # Save the aggregated statistics per layer to a JSON file
        with open(output_path, "w") as f:
            json.dump(all_layers, f, indent=4)
