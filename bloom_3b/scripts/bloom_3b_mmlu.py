import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import regex as re
import json

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model name short (used in filenames) and full HuggingFace model path
model_name_short = "bloom_3b"
model_name = "bigscience/bloom-3b"

# Define languages for evaluation and their language codes in mMMLU
languages_to_analyze_dict_mmlu = {
    "Croatian": "hr", "Russian": "ru",   # Slavic
    "Hindi": "hi",
    "English": "en", "German": "de",     # Germanic
    "Italian": "it", "Spanish": "es",    # Romance
    "Hungarian": "hu"                    # Other European
}

# Dictionary mapping answer letters to numerical positions
answers_dict = {"A": 1, "B": 2, "C": 3, "D": 4}

# Load model and tokenizer from Hugging Face with authentication token
access_token = "#####"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, output_hidden_states=True)
model.to(device)

# Format prompt for in-context evaluation of multiple-choice question
def format_prompt(question, choices):
    """
    Format the prompt with question and multiple choices for model input.
    """
    prompt = "Read the following question and potential answers."
    prompt += f" Question: {question}\n"
    prompt += "Answers:"
    prompt += f"\n1. {choices[0]}\n2. {choices[1]}\n3. {choices[2]}\n4. {choices[3]}"
    prompt += "\nAnswer only with the number of the correct answer. Only one answer is correct."
    prompt += "\nCorrect answer: "
    return prompt

# Extract numeric answer from model's text output
def extract_answer(response):
    """
    Extracts the predicted numeric answer from the model output string.
    """
    match = re.search(r"Correct answer:\s*(\S{1})", response)
    try:
        return int(match.group(1)) if match else 0
    except:
        return 0

# Dictionary to hold accuracy results for each language
languages_dict_results = {}

# Limit number of examples per language
number_of_examples = 250

# Loop through each language and evaluate
for language, language_key in languages_to_analyze_dict_mmlu.items():
    mmlu_language = load_dataset("alexandrainst/m_mmlu", language_key)["test"].to_pandas()
    correct = 0
    for index, row in mmlu_language.iterrows():
        if index == number_of_examples:
            break

        # Prepare question and choices
        question = row["instruction"]
        choices = [row["option_a"], row["option_b"], row["option_c"], row["option_d"]]
        prompt = format_prompt(question, choices)

        # Tokenize and run model
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20)

        # Decode and extract predicted answer
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = extract_answer(response)

        print(f"Answer: {answer}")
        print(f"Correct answer: {answers_dict[row['answer']]}")

        # Check correctness
        if int(answers_dict[row['answer']]) == int(answer):
            correct += 1

        if (index + 1) % (number_of_examples // 10) == 0:
            print(f"Evaluated {index + 1}/{number_of_examples} questions...")

    # Compute and store accuracy for current language
    languages_dict_results[language] = correct / number_of_examples
    print(f"{language} evaluated: {languages_dict_results[language]}")

# Save results to JSON file
output_path = f"{model_name_short}/mexa/task_results/{model_name_short}_mmlu_results.json"
with open(output_path, "w") as f:
    json.dump(languages_dict_results, f, indent=4)
