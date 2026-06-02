import pickle
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration
)

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# LOAD LABEL ENCODERS
# =====================================================

with open("type_label_encoder.pkl", "rb") as f:
    type_encoder = pickle.load(f)

with open("queue_label_encoder.pkl", "rb") as f:
    queue_encoder = pickle.load(f)

with open("priority_label_encoder.pkl", "rb") as f:
    priority_encoder = pickle.load(f)

# =====================================================
# LOAD MODELS
# =====================================================

# TYPE
type_tokenizer = AutoTokenizer.from_pretrained("type_model")
type_model = AutoModelForSequenceClassification.from_pretrained(
    "type_model"
).to(device)

# QUEUE
queue_tokenizer = AutoTokenizer.from_pretrained("queue_model")
queue_model = AutoModelForSequenceClassification.from_pretrained(
    "queue_model"
).to(device)

# PRIORITY
priority_tokenizer = AutoTokenizer.from_pretrained("priority_model")
priority_model = AutoModelForSequenceClassification.from_pretrained(
    "priority_model"
).to(device)

# ROOT CAUSE MODEL
root_tokenizer = T5Tokenizer.from_pretrained("root_cause_model")
root_model = T5ForConditionalGeneration.from_pretrained(
    "root_cause_model"
).to(device)

# =====================================================
# GENERIC CLASSIFICATION FUNCTION
# =====================================================

def classify(text, model, tokenizer, encoder):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    confidence = torch.softmax(
        outputs.logits,
        dim=1
    )[0][prediction].item()

    label = encoder.inverse_transform(
        [prediction]
    )[0]

    return label, round(confidence * 100, 2)


# =====================================================
# ROOT CAUSE GENERATION
# =====================================================

def generate_root_cause(text):

    prompt = f"generate root cause and solution: {text}"

    inputs = root_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    root_model.eval()

    with torch.no_grad():

        output_ids = root_model.generate(
            **inputs,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    result = root_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return result


# =====================================================
# MASTER FUNCTION
# =====================================================

def predict_ticket(ticket_text):

    ticket_type, type_conf = classify(
        ticket_text,
        type_model,
        type_tokenizer,
        type_encoder
    )

    queue, queue_conf = classify(
        ticket_text,
        queue_model,
        queue_tokenizer,
        queue_encoder
    )

    priority, priority_conf = classify(
        ticket_text,
        priority_model,
        priority_tokenizer,
        priority_encoder
    )

    root_cause = generate_root_cause(ticket_text)

    return {
        "Type": ticket_type,
        "Type Confidence": type_conf,

        "Queue": queue,
        "Queue Confidence": queue_conf,

        "Priority": priority,
        "Priority Confidence": priority_conf,

        "Root Cause & Resolution": root_cause
    }


# TEST

if __name__ == "__main__":

    text = """
    User unable to login after password reset.
    VPN access is not working.
    """

    result = predict_ticket(text)

    print(result)