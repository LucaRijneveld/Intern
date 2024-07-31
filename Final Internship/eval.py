import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the JSON data
with open(r'model/data/eval_Processed.json', 'r', encoding='utf-8') as file:
    test_data = json.load(file)

# Load label mappings
label_list = ['FAC', 'ACTIVITY', 'LOC', 'TIME', 'PRODUCT', 'PERSON', 'ORG', 'ABB', 'QUANTITY', 'DATE']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./results/BERT/base_uncased/00")

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

# Tokenize the texts and collect labels
test_tokenized_inputs = []
test_token_labels = []

print("Starting tokenization...")

for item in tqdm(test_data, desc="Tokenizing"):
    text = item['text']
    entities = item['unique_entities']
    
    inputs, labels = tokenize_and_label(text, entities)
    test_tokenized_inputs.append(inputs)
    test_token_labels.append(labels)

# Flatten test tokenized inputs
print("Flattening tokenized inputs...")
test_combined_inputs = {key: torch.cat([x[key] for x in test_tokenized_inputs], dim=0) for key in test_tokenized_inputs[0]}
test_combined_labels = torch.tensor([label + [-100] * (512 - len(label)) for label in test_token_labels])

# Verify the shapes of inputs and labels
print(f"Input shape: {test_combined_inputs['input_ids'].shape}")
print(f"Labels shape: {test_combined_labels.shape}")

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

# Create test dataset
test_dataset = NERDataset(test_combined_inputs, test_combined_labels)

# Load the pretrained model
model = AutoModelForTokenClassification.from_pretrained("./results/BERT/base_uncased/00", num_labels=len(label_list))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded. Beginning evaluation...")

def compute_metrics(predictions, labels, id2label):
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    valid_indices = labels != -100
    valid_labels = labels[valid_indices]
    valid_predictions = predictions[valid_indices]

    labels_present = sorted(set(valid_labels.tolist() + valid_predictions.tolist()))
    target_names = [id2label[i] for i in labels_present]

    report = classification_report(valid_labels, valid_predictions, labels=labels_present, target_names=target_names, output_dict=True)
    
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    return {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'report': report,
        'valid_labels': valid_labels,
        'valid_predictions': valid_predictions
    }

def plot_classification_metrics(report, metric, id2label, results_dir):
    labels = list(id2label.keys())
    metric_values = [report[id2label[label]][metric] if id2label[label] in report else 0 for label in labels]

    plt.figure(figsize=(10, 7))
    sns.barplot(x=[id2label[label] for label in labels], y=metric_values)
    plt.title(f'{metric.capitalize()} per class')
    plt.xlabel('Class')
    plt.ylabel(metric.capitalize())
    plt.ylim(0, 1)  # Metrics are in the range [0, 1]
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(results_dir, f"{metric}_per_class.png"))
    plt.close()

def evaluate_model(model, test_dataset, id2label, results_dir="./results/BERT/base_uncased/31"):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data in tqdm(test_dataset, desc="Evaluating"):
            inputs = {key: val.unsqueeze(0).to(device) for key, val in data.items() if key != 'labels'}
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=-1)

            predictions.append(predicted_labels.cpu().numpy())
            labels.append(data['labels'].unsqueeze(0).cpu().numpy())

    predictions = [item for sublist in predictions for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    
    metrics = compute_metrics(predictions, labels, id2label)
    print(f"Test Performance: {metrics}")

    metrics['valid_labels'] = metrics['valid_labels'].tolist()
    metrics['valid_predictions'] = metrics['valid_predictions'].tolist()
    
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f)

    with open(os.path.join(results_dir, "classification_report.json"), "w") as f:
        json.dump(metrics['report'], f, indent=4)
    
    valid_labels = np.array(metrics['valid_labels'])
    valid_predictions = np.array(metrics['valid_predictions'])
    conf_matrix = confusion_matrix(valid_labels, valid_predictions, labels=sorted(set(valid_labels.tolist())))
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=[id2label[i] for i in sorted(set(valid_labels.tolist()))], yticklabels=[id2label[i] for i in sorted(set(valid_labels.tolist()))])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))

    # Plot and save metrics for precision, recall, and F1-score
    plot_classification_metrics(metrics['report'], 'precision', id2label, results_dir)
    plot_classification_metrics(metrics['report'], 'recall', id2label, results_dir)
    plot_classification_metrics(metrics['report'], 'f1-score', id2label, results_dir)

    misclassified_examples = []
    for i, (true_label, pred_label) in enumerate(zip(valid_labels, valid_predictions)):
        if true_label != pred_label and i < len(test_dataset):
            misclassified_examples.append({
                'input_ids': test_dataset.encodings['input_ids'][i].tolist(),
                'true_label': id2label[true_label],
                'predicted_label': id2label[pred_label]
            })
    
    with open(os.path.join(results_dir, "misclassified_examples.json"), "w") as f:
        json.dump(misclassified_examples, f, indent=4)

# Run evaluation
evaluate_model(model, test_dataset, id2label)
