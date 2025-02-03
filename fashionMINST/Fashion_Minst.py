import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import time, copy
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# Datasets
fashion_mnist_train = torchvision.datasets.FashionMNIST('', train=True, transform=transform, download=True)
fashion_mnist_train, fashion_mnist_val = torch.utils.data.random_split(fashion_mnist_train, 
    [int(np.floor(len(fashion_mnist_train)*0.75)), int(np.ceil(len(fashion_mnist_train)*0.25))])
fashion_mnist_test = torchvision.datasets.FashionMNIST('', train=False, transform=transform, download=True)

# DataLoaders
batch_size = 100
dataloaders = {
    'train': DataLoader(fashion_mnist_train, batch_size=batch_size),
    'val': DataLoader(fashion_mnist_val, batch_size=batch_size),
    'test': DataLoader(fashion_mnist_test, shuffle=True, batch_size=batch_size)
}

# Dataset sizes
dataset_sizes = {
    'train': len(fashion_mnist_train),
    'val': len(fashion_mnist_val),
    'test': len(fashion_mnist_test)
}
print(f'Dataset sizes = {dataset_sizes}')

# CNN Classifier
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.pipeline(x)

# Hyperparameters
learning_rate = 0.001
num_epochs = 10
model = CNNClassifier().to(device)

# Training function
def train_classification_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    training_curves = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            training_curves[f'{phase}_loss'].append(epoch_loss)
            training_curves[f'{phase}_acc'].append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, training_curves

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
model, training_curves = train_classification_model(
    model, dataloaders, dataset_sizes, criterion, 
    optimizer, scheduler, num_epochs=num_epochs
)

# Plotting utilities
def plot_training_curves(training_curves, phases=['train', 'val'], metrics=['loss', 'acc']):
    epochs = list(range(len(training_curves['train_loss'])))
    for metric in metrics:
        plt.figure()
        plt.title(f'Training curves - {metric}')
        for phase in phases:
            key = phase+'_'+metric
            if key in training_curves:
                if metric == 'acc':
                    plt.plot(epochs, [item.detach().cpu() for item in training_curves[key]])
                else:
                    plt.plot(epochs, training_curves[key])
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend(labels=phases)
        plt.show()

def classify_predictions(model, device, dataloader):
    model.eval()
    all_labels = torch.tensor([]).to(device)
    all_preds = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels = torch.cat((all_labels, labels), 0)
            all_preds = torch.cat((all_preds, preds), 0)
    
    return all_preds.detach().cpu(), all_labels.detach().cpu()

def plot_cm(model, device, dataloaders, phase='test'):
    class_labels = list(range(10))
    preds, labels = classify_predictions(model, device, dataloaders[phase])
    
    cm = metrics.confusion_matrix(labels, preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

# Visualize results
plot_training_curves(training_curves, phases=['train', 'val'])
plot_cm(model, device, dataloaders, phase='test')