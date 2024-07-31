import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from sklearn.utils import resample
import numpy as np
import os
import random

# Load the JSON data
# Train data
with open(r'model/data/test_Processed.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Test data
with open(r'model/data/eval_Processed.json', 'r', encoding='utf-8') as file:
    test_data = json.load(file)
       
# Collect unique labels
unique_labels = set()
for item in data:
    for entity in item['unique_entities']:
        unique_labels.add(entity['label'])

label_list = list(unique_labels)
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

print("Label List:", label_list)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_label(text, entities):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    labels = [-100] * len(inputs['input_ids'][0])

    for entity in entities:
        entity_text = entity['text']
        entity_label = entity['label']
        entity_tokens = tokenizer.tokenize(entity_text)
        
        for i in range(len(inputs['input_ids'][0]) - len(entity_tokens) + 1):
            if tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][i:i + len(entity_tokens)]) == entity_tokens:
                labels[i:i + len(entity_tokens)] = [label2id[entity_label]] * len(entity_tokens)

    return inputs, labels

def hybrid_resample(data, label_key='unique_entities', target_size=None):
    # Create a dictionary to hold lists of examples for each class
    label_to_examples = {label: [] for label in label_list}
    
    # Populate the dictionary with examples
    for item in data:
        for entity in item[label_key]:
            label_to_examples[entity['label']].append(item)
            break  # Ensure each item is counted only once
    
    # Find the sizes of each class
    sizes = {label: len(examples) for label, examples in label_to_examples.items()}
    
    # Determine target size for balancing
    if target_size is None:
        target_size = min(sizes.values())  # Start with the smallest class size
    
    balanced_data = []
    
    # Undersample majority classes
    for label, examples in label_to_examples.items():
        if len(examples) > target_size:
            examples = resample(examples, n_samples=target_size, random_state=42)
        balanced_data.extend(examples)
    
    # Update sizes after undersampling
    sizes = {label: len([item for item in balanced_data if any(e['label'] == label for e in item[label_key])]) for label in label_list}
    
    # Oversample minority classes
    for label, examples in label_to_examples.items():
        if sizes[label] < target_size:
            minority_samples = [item for item in balanced_data if any(e['label'] == label for e in item[label_key])]
            minority_samples = resample(minority_samples, n_samples=target_size - sizes[label], random_state=42, replace=True)
            balanced_data.extend(minority_samples)
    
    random.shuffle(balanced_data)
    return balanced_data

# Apply hybrid resampling to your training data
balanced_data = hybrid_resample(data)

# Tokenize the texts and collect labels for balanced data
tokenized_inputs = []
token_labels = []

for item in balanced_data:
    text = item['text']
    entities = item['unique_entities']
    
    inputs, labels = tokenize_and_label(text, entities)
    tokenized_inputs.append(inputs)
    token_labels.append(labels)

# Tokenize the texts and collect labels for test data
test_tokenized_inputs = []
test_token_labels = []

for item in test_data:
    text = item['text']
    entities = item['unique_entities']
    
    inputs, labels = tokenize_and_label(text, entities)
    test_tokenized_inputs.append(inputs)
    test_token_labels.append(labels)

# Flatten tokenized inputs
combined_inputs = {key: torch.cat([x[key] for x in tokenized_inputs], dim=0) for key in tokenized_inputs[0]}
combined_labels = torch.tensor([label + [-100] * (512 - len(label)) for label in token_labels])

# Flatten test tokenized inputs
test_combined_inputs = {key: torch.cat([x[key] for x in test_tokenized_inputs], dim=0) for key in test_tokenized_inputs[0]}
test_combined_labels = torch.tensor([label + [-100] * (512 - len(label)) for label in test_token_labels])

# Verify the shapes of inputs and labels
print(f"Input shape: {combined_inputs['input_ids'].shape}")
print(f"Labels shape: {combined_labels.shape}")

# Define the dataset class
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset
dataset = NERDataset(combined_inputs, combined_labels)
test_dataset = NERDataset(test_combined_inputs, test_combined_labels)

# Initialize the model
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(label_list))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results/BERT/base_uncased/1',          # output directory
    num_train_epochs=10,              # number of training epochs
    per_device_train_batch_size=4,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./results/BERT/base_uncased/1")
tokenizer.save_pretrained("./results/BERT/base_uncased/1")

def compute_metrics(predictions, labels, id2label):
    # Flatten predictions and labels
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Ensure only non-pad tokens are evaluated
    valid_indices = labels != -100
    valid_labels = labels[valid_indices]
    valid_predictions = predictions[valid_indices]

    # Check the unique labels present in valid_labels and valid_predictions
    unique_valid_labels = np.unique(valid_labels)
    unique_valid_predictions = np.unique(valid_predictions)
    
    # Ensure that the labels parameter in classification_report matches the actual labels present in the data
    labels_present = sorted(set(unique_valid_labels.tolist() + unique_valid_predictions.tolist()))
    target_names = [id2label[i] for i in labels_present]

    # Calculate precision, recall, and F1 score for each label
    report = classification_report(valid_labels, valid_predictions, labels=labels_present, target_names=target_names, output_dict=True)
    
    # Overall weighted metrics
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    return {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'report': report  # Include full report for per-label metrics
    }

def evaluate_model(model, test_dataset, id2label, results_dir="./results/BERT/base_uncased/1"):
    model.eval()  # Set model to evaluation mode
    predictions = []
    labels = []

    with torch.no_grad():
        for data in test_dataset:
            # Prepare inputs and ensure they are on the same device as the model
            inputs = {key: val.unsqueeze(0).to(model.device) for key, val in data.items() if key != 'labels'}
            outputs = model(**inputs)  # Forward pass
            logits = outputs.logits  # Get logits from model output
            predicted_labels = torch.argmax(logits, dim=-1)  # Get predicted labels

            predictions.append(predicted_labels.cpu().numpy())
            labels.append(data['labels'].unsqueeze(0).cpu().numpy())

    # Flatten predictions and labels
    predictions = [item for sublist in predictions for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    
    metrics = compute_metrics(predictions, labels, id2label)  # Replace with your evaluation metric function
    print(f"Test Performance: {metrics}")
    
    # Save metrics to a file
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "run00_moredata_metrics.json"), "w") as f:
        json.dump(metrics, f)

    # Save detailed classification report
    with open(os.path.join(results_dir, "classification_report.json"), "w") as f:
        json.dump(metrics['report'], f, indent=4)

# Evaluate the model
evaluate_model(model, test_dataset, id2label)
