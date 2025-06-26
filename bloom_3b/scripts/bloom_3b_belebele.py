from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import regex as re

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model identifier and Hugging Face model repo name
model = "bloom_3b"  # Short model name used for saving paths
model_name = "bigscience/bloom-3b"  # Hugging Face model repo
access_token = "#####"

# Load tokenizer and model with hidden state outputs enabled
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, output_hidden_states=True)
model.to(device)  # Move model to GPU

# Languages from Belebele dataset to evaluate
languages_to_analyze_dict_belebele = {
    "Swahili": "swh_Latn", "Yoruba": "yor_Latn",  # African Languages
    "Hindi": "hin_Deva", "Urdu": "urd_Arab",      # Indic Languages
    "English": "eng_Latn", "German": "deu_Latn",  # Germanic Languages
    "Italian": "ita_Latn", "Spanish": "spa_Latn",  # Romance Languages
    "Polish": "pol_Latn", "Croatian": "hrv_Latn", "Russian": "rus_Cyrl",  # Slavic Languages
    "Hungarian": "hun_Latn",  # Other European Language
}

def format_prompt(question, choices, context):
    """
    Formats a multiple-choice prompt with numbered choices and clear instruction.
    """
    prompt = f"Context: {context}\n"
    prompt += f"Question: {question}\n"
    prompt += "\n".join([f"{i+1}) {choice}" for i, choice in enumerate(choices)])
    prompt += "\nAnswer only with the number of the correct answer. Only one answer is correct."
    prompt += "\nAnswer: "
    return prompt

def extract_answer(response):
    """
    Extracts numerical answer (1-4) from model's generated response.
    Returns 0 if extraction fails.
    """
    match = re.search(r"Answer:\s*(\S{1})", response)
    try:
        return int(match.group(1)) if match else 0
    except:
        return 0

def evaluate_question(question, choices, context):
    """
    (Optional function) Evaluates which choice has the highest log-probability
    – currently not used in the main loop.
    """
    prompt = format_prompt(question, choices, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Log-likelihood estimation per choice
    scores = []
    for i, choice in enumerate(choices):
        choice_inputs = tokenizer(f"{prompt}{i+1}", return_tensors="pt").to(device)
        with torch.no_grad():
            choice_outputs = model(**choice_inputs)
        scores.append(choice_outputs.logits[:, -1].mean().item())

    # Select the choice with the highest score
    best_choice = scores.index(max(scores)) + 1
    return int(best_choice)

# Dictionary to store accuracy scores per language
languages_dict_results = {}

# Evaluate each language subset in the Belebele dataset
for language, language_key in languages_to_analyze_dict_belebele.items():
    correct = 0
    # Load test split of the dataset
    belebele_language = load_dataset("facebook/belebele", language_key)["test"].to_pandas()

    # Iterate through the first 100 questions
    for index, row in belebele_language.iterrows():
        if index == 100:
            break
        question = row["question"]
        choices = [row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]]
        context = row["flores_passage"]

        # Format prompt and run model generation
        prompt = format_prompt(question, choices, context)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=5)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = extract_answer(response)

        print(f"Answer: {answer}")
        print(f"Correct answer: {row['correct_answer_num']}")

        if int(row["correct_answer_num"]) == int(answer):
            correct += 1

        if (index + 1) % 10 == 0:
            print(f"Evaluated {index + 1}/100 questions...")

    # Save accuracy result for this language
    languages_dict_results[language] = correct / 100
    print(f"{language} evaluated: {languages_dict_results[language]}")

# Save evaluation results as JSON
output_path = f"{model}/mexa/task_results/{model}_belebele_results.json"
with open(output_path, "w") as f:
    json.dump(languages_dict_results, f, indent=4)
