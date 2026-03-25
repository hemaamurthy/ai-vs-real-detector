from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# classifier model
clf_name = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(clf_name)
model = AutoModelForSequenceClassification.from_pretrained(clf_name)

# language model for perplexity
lm_name = "gpt2"
lm_tokenizer = AutoTokenizer.from_pretrained(lm_name)
lm_model = AutoModelForCausalLM.from_pretrained(lm_name)


def detect_text(text):
    # classification
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    ai_prob = probs[0][1].item()

    # perplexity
    enc = lm_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = lm_model(**enc, labels=enc["input_ids"]).loss

    perplexity = torch.exp(loss).item()

    if ai_prob > 0.7 and perplexity < 80:
     final = "AI Generated"
    elif ai_prob < 0.3 and perplexity > 100:
     final = "Human Written"
    else:
     final = "Uncertain"

    return {
    "result": final,
    "ai_probability": round(ai_prob * 100, 2),
    "perplexity": round(perplexity, 2)
}