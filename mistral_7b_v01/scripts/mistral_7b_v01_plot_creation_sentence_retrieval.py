import json
import matplotlib.pyplot as plt

# Set model name as a variable for reusability
model = "mistral_7b_v01"

# Number of runs and data examples per run
number_of_iterations = 5
number_of_examples = 100

# Different embedding pooling strategies used for comparison
options = ["all", "last", "weighted"]

# Run over multiple iterations
for iteration in range(number_of_iterations):

    file_paths = []

    # Construct file paths for the current iteration and each option
    for option in options:
        file_paths.append(
            f"{model}/percentages_of_most_similar_representations_same_example_and_language/{option}/"
            f"{model}_percentages_most_similar_retrieved_representations_{iteration+1}_iteration_"
            f"{number_of_examples}_examples_top_100_{option}.json"
        )

    # Set up subplots: one for similarity percentage, one for average rank
    fig, axes = plt.subplots(2, 1, figsize=(10, 13))

    # Colors for plotting lines in each subplot
    first_plot_colors = ['b', 'g', 'r', 'c', 'm', 'y']
    second_plot_colors = ['b', 'g', 'r', 'c', 'm', 'y']
    labels = options

    # Plot first two keys (typically percentages) in the first subplot
    color_idx = 0
    for file_idx, file_path in enumerate(file_paths):
        with open(file_path, "r") as f:
            retrieved = json.load(f)
        
        x_labels = list(retrieved.keys())  # Usually layers
        sub_keys = list(next(iter(retrieved.values())).keys())
        first_two_keys = sub_keys[:2]

        for key in first_two_keys:
            values = [retrieved[x][key] for x in x_labels]
            axes[0].plot(
                x_labels, values, marker='o', linestyle='-',
                label=f"{labels[file_idx]} - {key}",
                color=first_plot_colors[color_idx]
            )
            color_idx += 1

    # Plot last two keys (typically average ranks) in the second subplot
    color_idx = 0
    for file_idx, file_path in enumerate(file_paths):
        with open(file_path, "r") as f:
            retrieved = json.load(f)
        
        x_labels = list(retrieved.keys())
        sub_keys = list(next(iter(retrieved.values())).keys())
        last_two_keys = sub_keys[2:]

        for key in last_two_keys:
            values = [retrieved[x][key] for x in x_labels]
            axes[1].plot(
                x_labels, values, marker='o', linestyle='-',
                label=f"{labels[file_idx]} - {key}",
                color=second_plot_colors[color_idx]
            )
            color_idx += 1

    # Customize first subplot: similarity percentages
    axes[0].set_title("Percentage of most similar retrieved representations for same language and same meaning")
    axes[0].set_xlabel("Layers")
    axes[0].set_ylabel("Percentage")
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)

    # Customize second subplot: average similarity ranks
    axes[1].set_title("Average place in similarity ranking for same language and same meaning")
    axes[1].set_xlabel("Layers")
    axes[1].set_ylabel("Average place")
    axes[1].legend()
    axes[1].set_ylim(0, 1200)
    axes[1].grid(alpha=0.3)

    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(f"{model}/plots_retrieval/{model}_retrieval_{iteration+1}_iteration_{number_of_examples}_examples.png", bbox_inches='tight')
    plt.show()
