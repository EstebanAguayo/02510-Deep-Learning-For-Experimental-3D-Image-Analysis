import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

class Config:
    def __init__(self, input_size, layer1_size, layer2_size, output_size):
        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.output_size = output_size

###################
# HYPERPARAMETERS:#
###################
BATCH_SIZE=32
EPOCHS = 2
LR = 0.001


Transform = transforms.Compose([
    transforms.ToTensor()
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ])

train_set = torchvision.datasets.MNIST('./data', train=True, download=True, transform=Transform)
test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=Transform)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

class Classifier(torch.nn.Module):
    
    def __init__(self,config):
        super(Classifier,self).__init__()
        self.layer1 = torch.nn.Linear(config.input_size, config.layer1_size)
        self.layer2 = torch.nn.Linear(config.layer1_size, config.layer2_size)
        self.layer3 = torch.nn.Linear(config.layer2_size, config.output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
config = Config(input_size=784, layer1_size=128, layer2_size=64, output_size=10)
model = Classifier(config)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)

global_loss = []
accuracy = []

global_train_loss = []
train_accuracy = []
global_test_loss = []
test_accuracy = []
EPOCHS = 2

for i in range(EPOCHS):
    epoch_train_loss = []
    epoch_train_accuracy = []
    epoch_test_loss = []
    epoch_test_accuracy = []
    print(f"Epoch {i+1} of {EPOCHS}")

    # TRAINING
    model.train()
    for j, batch in enumerate(trainloader):
        #Unpack the data
        images = batch[0].view(len(batch[0]),-1)
        labels = batch[1]

        #Feed the data
        outputs = model(images)
        optimizer.zero_grad()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        #Store statistical data
        epoch_train_loss.append(loss.item())
        batch_accuracy = (torch.argmax(outputs, dim=1) == labels).float().mean().item()
        epoch_train_accuracy.append(batch_accuracy)
    print(f"Completed Training. Loss: {loss.item():.3f} Accuracy: {batch_accuracy:.3f}")

    # EVALUATING
    model.eval()
    for j, batch in enumerate(testloader):
        #Unpack the data
        images = batch[0].view(len(batch[0]),-1)
        labels = batch[1]

        #Feed the data
        outputs = model(images)
        loss = loss_function(outputs, labels)

        #Store statistical data
        epoch_test_loss.append(loss.item())
        batch_test_accuracy = (torch.argmax(outputs, dim=1) == labels).float().mean().item()
        epoch_test_accuracy.append(batch_test_accuracy)

    print(f"Completed Evaluation. Loss: {loss.item():.3f} Accuracy: {batch_test_accuracy:.3f}")
    
    # Compute and store epoch statistics
    global_train_loss.append(sum(epoch_train_loss) / len(epoch_train_loss))
    train_accuracy.append(sum(epoch_train_accuracy) / len(epoch_train_accuracy))
    global_test_loss.append(sum(epoch_test_loss) / len(epoch_test_loss))
    test_accuracy.append(sum(epoch_test_accuracy) / len(epoch_test_accuracy))