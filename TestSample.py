from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load BART MNLI model
model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def fact_check(claim, passage):
    # Encode claim + passage pair
    inputs = tokenizer(claim, passage, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    # Softmax probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    # Labels used by MNLI
    labels = ["contradiction (False)", "neutral (Not Enough Info)", "entailment (True)"]
    
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Example usage
claim = "The sun rises in the east."
passage = "The Earth rotates from west to east, so the sun appears to rise in the east."

print(fact_check(claim, passage))
