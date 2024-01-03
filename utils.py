from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
import matplotlib.pyplot as plt

def download_MNIST():

    train = datasets.MNIST(os.getcwd(), train=True, download=True, transform = transforms.ToTensor()) 
    test = datasets.MNIST(os.getcwd(), train=False, download=True, transform = transforms.ToTensor())
        
    return train, test

def data_loaders(train, test, batch_size):
    train_dataloader=DataLoader(dataset=train, batch_size=batch_size,shuffle=True)
    test_dataloader=DataLoader(dataset=test, batch_size=batch_size,shuffle=True)
    return train_dataloader, test_dataloader

def accuracy(outputs, labels):
    # total_instances = len(outputs)
    
    # predictions = torch.argmax(outputs, dim=1)
    # correct_predictions = sum(predictions==labels).item()
    
    return labels.eq(outputs.detach().argmax(dim=1)).float().mean()

def adam_optimizer(model, lr): 
    return torch.optim.Adam(model.parameters(), lr=lr)
    
def cross_entropy_loss():
    return torch.nn.CrossEntropyLoss()

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs):

    log_dict = {
        'train_loss_per_epoch' :  [],
        'train_accuracy_per_epoch' : [],
        'test_loss_per_epoch' :  [],
        'test_accuracy_per_epoch' :  [],
    }
        
    for epoch in range(num_epochs):
        print(f'Epoch : {epoch + 1}/{num_epochs}')
        
        running_loss_train = list()
        running_accuracy_train = list()
        
        loss_test = list()
        accuracy_test = list()
        
        # Training 
        for i, batch in enumerate(train_loader, 0):
            inputs, labels = batch
            
            # inputs of size (batch_size, 1, 28 * 28)
            # inputs = inputs.view(inputs.size(0), -1) uncomment it if you're using FCN
            
            # 1 : Forward 
            outputs = model(inputs)     
            
            # 2 : Loss Computation
            loss = criterion(outputs, labels)
            
            # 3 : Cleaning the gradients
            optimizer.zero_grad()
            
            # 4 : Accumulate the partial derivative of the loss wrt to the params
            loss.backward()
            
            # 5 : Step in the opposite direction of the gradient
            optimizer.step()
            
            
            running_loss_train.append(loss.item()) 
            running_accuracy_train.append(accuracy(outputs, labels))
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {(loss.item()):.3f}, accuracy: {(accuracy(outputs, labels)):.3f}', end="\r", flush=True)
        
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {(torch.tensor(running_loss_train).mean()):.3f}, accuracy: {(torch.tensor(running_accuracy_train).mean()):.3f}')  
        
        # Testing 
        model.eval()
        for inputs, labels in test_loader:
            with torch.no_grad():
                # inputs = inputs.view(inputs.size(0), -1) uncomment it if you're using FCN
                outputs = model(inputs)
                
            loss = criterion(outputs, labels)
            loss_test.append(loss.item()) 
            accuracy_test.append(accuracy(outputs, labels))

        log_dict['train_loss_per_epoch'].append(torch.tensor(running_loss_train).mean())
        log_dict['train_accuracy_per_epoch'].append(torch.tensor(running_accuracy_train).mean())
        log_dict['test_loss_per_epoch'].append(torch.tensor(loss_test).mean())
        log_dict['test_accuracy_per_epoch'].append(torch.tensor(accuracy_test).mean())
            
    print("Training complete.")
    return model, log_dict  

def visualization(train_losses, train_accuracy, val_losses, val_accuracy):
    epochs = range(len(train_losses))
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs - Folds')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

