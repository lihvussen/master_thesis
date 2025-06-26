import json

# Pooling/embedding options
options = ["all", "last", "weighted"]

# List of tasks for which alignment results will be processed
tasks = ["mexa", "cosine_alignment", "retrieval_alignment"]

# Model name used in file path and output file naming
model = "gemma_7b"

# Number of top and bottom language pairs to extract
number = 10

# Loop over each task
for task in tasks:
    task_dict = {}  # Dictionary to store top and low results for all options for the current task

    # Loop over each pooling option
    for option in options:
        option_dict = {}  # Dictionary to store top/low pairs for each iteration and pooling type

        try:
            # Try loading the alignment results file for the full task name
            with open(f"{model}/{task}/alignment_results/{model}_alignment_results_{option}_{task}.json", "r") as f:
                results = json.load(f)
        except:
            # If file not found, fallback to base task name (e.g., "cosine" from "cosine_alignment")
            new_task = task.split("_")[0]
            with open(f"{model}/{task}/alignment_results/{model}_alignment_results_{option}_{new_task}.json", "r") as f:
                results = json.load(f)

        # Process each iteration and pooling type in the loaded result
        for iteration, poolings in results.items():
            for pooling, language_pairs in poolings.items():
                # Sort language pairs by score in descending order
                sorted_items = sorted(language_pairs.items(), key=lambda item: item[1], reverse=True)

                # Take top N language pairs with highest scores
                top_dict = dict(sorted_items[:number])

                # Take bottom N language pairs with lowest scores
                low_dict = dict(sorted_items[-number:])

                # Store top and low results under a key identifying iteration, pooling type, and rank
                option_dict[f"{iteration}_{pooling}_top_{number}"] = top_dict
                option_dict[f"{iteration}_{pooling}_low_{number}"] = low_dict

        # Save results for the current option under the main task dictionary
        task_dict[option] = option_dict

    # Write top and bottom results for the task to a JSON file
    with open(f"{model}/{task}/alignment_results/{model}_alignment_results_top_{number}s.json", "w") as f:
        json.dump(task_dict, f, indent=4, ensure_ascii=False)