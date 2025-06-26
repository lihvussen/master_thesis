import matplotlib.pyplot as plt
import json
import numpy as np

# Set the model name as a variable for easy reuse
model = "gemma_7b"

# List of tasks to process
tasks = ["mmlu", "belebele", "translation"]

# Loop through each task and generate a bar chart
for task in tasks:
    # Load JSON results for the current task
    with open(f"{model}/mexa/task_results/{model}_{task}_results.json", "r") as f:
        task_results = json.load(f)

    # Extract language labels and their corresponding scores
    labels = list(task_results.keys())
    values = list(task_results.values())

    # Set a clean and readable style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Generate a colormap with enough distinct colors for all bars
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

    # Create a new figure for the current task
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=colors, edgecolor='black')

    # Set plot labels and title
    plt.xlabel('Languages', fontsize=12)
    plt.ylabel('Scores', fontsize=12)
    plt.title(f'Language Scores for {task} Task', fontsize=14, fontweight='bold')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    # Set y-axis limits (assuming scores are between 0 and 1)
    plt.ylim(0, 1)

    # Add horizontal grid lines for better visual alignment
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Save the plot as a PNG image in the corresponding directory
    plt.savefig(f"{model}/mexa/task_results/{model}_{task}_plot.png", bbox_inches='tight')
    
    # Display the plot
    plt.show()
