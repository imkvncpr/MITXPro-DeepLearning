from io import open
import unicodedata
import string
import random
import re
import torch
import torch.nn as nn
import numpy as np
import time, copy
import matplotlib.pyplot as plt
import os
import requests
import zipfile
import io

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_dataset():
    url = "https://www.manythings.org/anki/spa-eng.zip"
    
    if not os.path.exists("spa.txt"):
        print("Downloading Spanish-English dataset...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print("Download successful. Extracting files...")
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall()
            if os.path.exists("_about.txt"):
                os.remove("_about.txt")
            print("Dataset ready!")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
    else:
        print("Dataset already exists!")

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-zA-Z.!'?]+", r" ", s)
    return s

def parse_data(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [[pair[0], pair[1]] for pair in pairs]
    return pairs

def add_words_to_dict(word_dictionary, word_list, sentences):
    for sentence in sentences:
        for word in sentence.split(" "):
            if word not in word_dictionary:
                word_list.append(word)
                word_dictionary[word] = len(word_list)-1

def create_input_tensor(sentence, word_dictionary):
    words = sentence.split(" ")
    tensor = torch.zeros(len(words), 1, len(word_dictionary)+1)
    for idx in range(len(words)):
        word = words[idx]
        if word in word_dictionary:
            tensor[idx][0][word_dictionary[word]] = 1
    return tensor

def create_target_tensor(sentence, word_dictionary):
    words = sentence.split(" ")
    # Create tensor with proper dimensions for NLLLoss
    tensor = torch.zeros(len(words), len(word_dictionary)+1)
    for idx in range(1, len(words)):
        word = words[idx]
        if word in word_dictionary:
            tensor[idx-1][word_dictionary[word]] = 1
    tensor[len(words)-1][len(word_dictionary)] = 1  # EOS
    return tensor

def tensor_to_sentence(word_list, tensor):
    sentence = ""
    for i in range(tensor.size(0)):
        topv, topi = tensor[i].topk(1)
        if topi[0][0] == len(word_list):
            sentence += "<EOS>"
            break
        sentence += word_list[topi[0][0]]
        sentence += " "
    return sentence

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM layer with 2 layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=False)
        
        # Output layers
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        # Process through LSTM
        output, hidden = self.lstm(input, hidden)
        
        # Pass through final layers
        output = self.fc(output.view(-1, self.hidden_size))
        output = self.softmax(output)
        
        return output, hidden
    
    def initHidden(self):
        return (torch.zeros(2, 1, self.hidden_size).to(device),
                torch.zeros(2, 1, self.hidden_size).to(device))

def train_lstm(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_epoch = 0
    phases = ['train', 'val', 'test']
    
    training_curves = {phase+'_loss': [] for phase in phases}
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for input_sequence, target_sequence in dataloaders[phase]:
                hidden = model.initHidden()
                
                current_input_sequence = input_sequence.to(device)
                current_target_sequence = target_sequence.to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss = 0
                    for i in range(current_input_sequence.size(0)):
                        output, hidden = model(current_input_sequence[i:i+1], hidden)
                        # Convert target to proper format for NLLLoss
                        target = current_target_sequence[i].argmax(dim=0).unsqueeze(0)
                        loss += criterion(output, target)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() / current_input_sequence.size(0)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            training_curves[phase+'_loss'].append(epoch_loss)

            print(f'{phase:5} Loss: {epoch_loss:.4f}')

            if phase == 'train' and epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:4f} at epoch {best_epoch}')

    model.load_state_dict(best_model_wts)
    return model, training_curves

def predict(model, word_dictionary, word_list, input_sentence, max_length=20):
    model.eval()
    output_sentence = input_sentence + " "
    tensor = create_input_tensor(input_sentence, word_dictionary)
    hidden = model.initHidden()
    
    with torch.no_grad():
        # Process input sentence
        for i in range(tensor.size(0)):
            output, hidden = model(tensor[i:i+1].to(device), hidden)
            
        # Generate new words
        current_length = len(input_sentence.split())
        for _ in range(max_length - current_length):
            topv, topi = output.topk(1)
            word_idx = topi[0].item()
            
            if word_idx == len(word_dictionary):  # EOS token
                break
                
            word = word_list[word_idx]
            output_sentence += word + " "
            
            # Prepare input for next iteration
            next_input = create_input_tensor(word, word_dictionary)
            output, hidden = model(next_input[0:1].to(device), hidden)
    
    return output_sentence.strip()

def plot_training_curves(training_curves, phases=['train', 'val', 'test'], metrics=['loss']):
    epochs = list(range(len(training_curves['train_loss'])))
    for metric in metrics:
        plt.figure()
        plt.title(f'Training curves - {metric}')
        for phase in phases:
            key = phase+'_'+metric
            if key in training_curves:
                plt.plot(epochs, training_curves[key])
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend(labels=phases)
        plt.show()

# Main execution
if __name__ == "__main__":
    # Download dataset
    download_dataset()
    
    # Process the data
    pairs = parse_data("spa.txt")
    english_sentences = [pair[0] for pair in pairs]
    random.shuffle(english_sentences)
    print("Number of English sentences:", len(english_sentences))

    # Split the dataset
    train_sentences = english_sentences[:1000]
    val_sentences = english_sentences[1000:2000]
    test_sentences = english_sentences[2000:3000]

    # Create vocabulary
    english_dictionary = {}
    english_list = []
    add_words_to_dict(english_dictionary, english_list, train_sentences)
    add_words_to_dict(english_dictionary, english_list, val_sentences)
    add_words_to_dict(english_dictionary, english_list, test_sentences)

    # Create tensors
    train_tensors = [(create_input_tensor(sentence, english_dictionary), 
                      create_target_tensor(sentence, english_dictionary)) 
                     for sentence in train_sentences]
    val_tensors = [(create_input_tensor(sentence, english_dictionary), 
                    create_target_tensor(sentence, english_dictionary)) 
                   for sentence in val_sentences]
    test_tensors = [(create_input_tensor(sentence, english_dictionary), 
                     create_target_tensor(sentence, english_dictionary)) 
                    for sentence in test_sentences]

    # Create dataloaders
    dataloaders = {
        'train': train_tensors,
        'val': val_tensors,
        'test': test_tensors
    }

    dataset_sizes = {
        'train': len(train_tensors),
        'val': len(val_tensors),
        'test': len(test_tensors)
    }

    # Model parameters
    input_size = len(english_dictionary) + 1
    hidden_size = 256
    output_size = len(english_dictionary) + 1

    # Initialize model, loss, optimizer
    lstm = LSTM(input_size, hidden_size, output_size).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    num_epochs = 25
    lstm, training_curves = train_lstm(lstm, dataloaders, dataset_sizes, 
                                     criterion, optimizer, scheduler, num_epochs=num_epochs)

    # Test some predictions
    test_phrases = ["what is", "my name", "how are", "hi", "choose"]
    print("\nExample predictions:")
    for phrase in test_phrases:
        prediction = predict(lstm, english_dictionary, english_list, phrase)
        print(f"Input: {phrase}")
        print(f"Output: {prediction}\n")

    # Plot training curves
    plot_training_curves(training_curves, phases=['train', 'val', 'test'])