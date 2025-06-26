import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

models = ["aya_8b", "bloom_3b", "deepseek_7b_base", "gemma_7b", "llama2_7b", "llama3_1_8b", "mistral_7b_v01"]
tasks = ["belebele", "mmlu", "translation"]

# Store results as a list of dicts
data = []

for model in models:
    for task in tasks:
        file_path = os.path.join(model, "mexa", "task_results", f"{model}_{task}_results.json")
        try:
            with open(file_path, "r") as f:
                scores = json.load(f)
                avg_score = sum(scores.values()) / len(scores)
                data.append({"Model": model, "Task": task, "Score": avg_score})
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Create DataFrame
df = pd.DataFrame(data)

# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Create barplot
ax = sns.barplot(data=df, x="Model", y="Score", hue="Task", palette="Set2")

# Format plot
ax.set_title("Average Task Scores per Model")
ax.set_ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(title="Task")
plt.tight_layout()

# Save the plot
output_dir = "general_results"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "plot_all_languages_merged.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Plot saved to {plot_path}")