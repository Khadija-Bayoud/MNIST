from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os

def download_MNIST():

    train = datasets.MNIST(os.getcwd(), train=True, download=True, transform = transforms.ToTensor()) 
    test = datasets.MNIST(os.getcwd(), train=False, download=True, transform = transforms.ToTensor())
        
    return train, test

def data_loaders(train, test, batch_size):
    train_dataloader=DataLoader(dataset=train, batch_size=batch_size,shuffle=True)
    test_dataloader=DataLoader(dataset=test, batch_size=batch_size,shuffle=True)
    return train_dataloader, test_dataloader

def accuracy(outputs, labels):
    total_instances = len(outputs)
    
    predictions = torch.argmax(outputs, dim=1)
    correct_predictions = sum(predictions==labels).item()
    
    return round(correct_predictions/total_instances, 3)

def adam_optimizer(model, lr): 
    return torch.optim.Adam(model.parameters(), lr=lr)
    
def cross_entropy_loss():
    return torch.nn.CrossEntropyLoss()

def train_model(model, train_loader, optimizer, criterion, num_epochs):

    log_dict = {
        'train_loss_per_epoch' :  [],
        'train_accuracy_per_epoch' : [],
    }
        
    for epoch in range(num_epochs):
        print(f'Epoch : {epoch + 1}/{num_epochs}')
        
        running_loss = 0.0
        running_accuracy = 0.0
        
        for i, batch in enumerate(train_loader, 0):
            inputs, labels = batch
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)       
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_accuracy += accuracy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {(loss.item()):.3f}, accuracy: {(accuracy(outputs, labels)):.3f}', end="\r", flush=True)
                      
        train_loss = running_loss/len(train_loader)
        train_accuracy = running_accuracy/len(train_loader) 
        
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {(loss.item()):.3f}, accuracy: {(accuracy(outputs, labels)):.3f}')  
        
        log_dict['train_loss_per_epoch'].append(train_loss)
        log_dict['train_accuracy_per_epoch'].append(train_accuracy)
            
    print("Training complete.")
    return model, log_dict  

def test_model(model, test_loader, criterion):  
    total_loss = 0.0
    total_accuracy = 0.0
    total_instances = 0

    with torch.no_grad():  
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_accuracy += accuracy(outputs, labels)
            total_instances += len(inputs)

    average_loss = total_loss / total_instances
    average_accuracy = total_accuracy / total_instances
    return average_loss, average_accuracy 
