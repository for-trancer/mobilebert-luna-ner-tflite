import json
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# -------------------------------
# 1. Load the Label Mapping & Tokenizer
# -------------------------------
with open("./mobilebert-luna-ner-tflite/intent_mapping.json", "r") as f:
    id2label_mapping = json.load(f)

model_dir = "./mobilebert-luna-ner"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# -------------------------------
# 2. Load the TFLite Model
# -------------------------------
tflite_model_file = "mobilebert-luna-ner-tflite/model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite Model Input Details:")
for inp in input_details:
    print("  Name:", inp["name"], "Shape:", inp["shape"], "Dtype:", inp["dtype"])

print("\nTFLite Model Output Details:")
for out in output_details:
    print("  Name:", out["name"], "Shape:", out["shape"], "Dtype:", out["dtype"])

# -------------------------------
# 3. Define 20 Test Examples
# -------------------------------
test_examples = [
    "set an alarm for 4:00 ",
    "play some music",
    "what's the weather like in New York",
    "remind me to call mom at 5 pm",
    "turn off the living room lights",
    "set a timer for 10 minutes",
    "open the door",
    "increase the temperature by 3 degrees",
    "what time is it",
    "what is my schedule today",
    "send a message to John saying I'll be late",
    "start the coffee machine",
    "what are the latest news headlines",
    "navigate to the nearest gas station",
    "order a pizza",
    "find me a nearby restaurant",
    "turn on the air conditioning",
    "cancel my meeting at 3 pm",
    "book a cab for tomorrow morning",
    "what is the exchange rate for USD to EUR"
]

# -------------------------------
# 4. Run Inference for Each Example and Print Results
# -------------------------------
for sentence in test_examples:
    print("\nProcessing sentence:", sentence)
    
    # Tokenize input with offsets.
    inputs = tokenizer(
        sentence,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")
    
    # Set TFLite inputs.
    for inp in input_details:
        if "input_ids" in inp["name"]:
            input_data = inputs["input_ids"]
        elif "attention_mask" in inp["name"]:
            input_data = inputs["attention_mask"]
        else:
            input_data = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
        if input_data.dtype != inp["dtype"]:
            input_data = input_data.astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], input_data)
    
    interpreter.invoke()
    
    # Retrieve model output (logits) with expected shape: [1, seq_length, num_labels]
    logits = interpreter.get_tensor(output_details[0]["index"])
    print("Logits shape:", logits.shape)
    
    # Process output: apply softmax, get predictions.
    probs = softmax(logits, axis=-1)
    predictions = np.argmax(logits, axis=-1)[0]  # Shape: (seq_length,)
    predicted_probs = probs[0]                    # Shape: (seq_length, num_labels)
    
    # Convert input_ids back to tokens.
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    results = []
    # Loop over tokens, skipping special tokens and padding.
    for i, (token, pred_id, prob_vector, offset) in enumerate(zip(tokens, predictions, predicted_probs, offset_mapping[0])):
        if token in tokenizer.all_special_tokens or offset[0] == offset[1]:
            continue
        label = id2label_mapping.get(str(pred_id), "Unknown")
        if label == "O":
            continue
        score = float(prob_vector[pred_id])
        results.append({
            "word": token,
            "entity": label,
            "score": score,
            "index": int(i),
            "start": int(offset[0]),
            "end": int(offset[1])
        })
    
    print("NER Results:")
    print(json.dumps(results, indent=4))
