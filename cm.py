import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load intents.json and preprocess data
with open("intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents["intenst"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", ".", "!"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for pattern_sentence, tag in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
num_epochs = 1000
batch_size = 32
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 32
output_size = len(tags)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

total_samples += labels.size(0)
total_correct += (predicted == labels).sum().item()

if (epoch + 1) % 100 == 0:
    accuracy = total_correct / total_samples  # Calculate accuracy here
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}"
    )

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"training complete. file saved to {FILE}")

# Prepare the model for inference
model.eval()


# Define a function to get the model's response
def get_response(model, all_words, tags, input_text):
    # Tokenize and process the input
    input_text = tokenize(input_text)
    input_text = [stem(word) for word in input_text]
    bag = bag_of_words(input_text, all_words)
    bag = torch.tensor(bag, dtype=torch.float32).to(device)
    bag = bag.unsqueeze(0)

    # Get the model's output
    output = model(bag)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    return tag


# Prepare data for confusion matrix
true_labels = []
predictions = []

for intent in intents["intenst"]:
    tag = intent["tag"]
    for example in intent["patterns"]:
        true_labels.append(tag)
        predicted_tag = get_response(model, all_words, tags, example)
        predictions.append(predicted_tag)

# Calculate and display confusion matrix
confusion = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(confusion)
import matplotlib.pyplot as plt
import seaborn as sns


# Define a function to plot the confusion matrix
def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# Call the function to plot the confusion matrix
plot_confusion_matrix(confusion, class_names=tags)

# Calculate and display classification report
classification_rep = classification_report(true_labels, predictions, target_names=tags)
print("Classification Report:")
print(classification_rep)
