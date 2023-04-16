import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl


###################
# HYPERPARAMETERS:#
###################
BATCH_SIZE = 32
EPOCHS = 2
LR = 0.001

class Config:
    def __init__(self, input_size, layer1_size, layer2_size, output_size, lr):
        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.output_size = output_size
        self.lr = lr

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

class MNISTClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.layer1 = nn.Linear(config.input_size, config.layer1_size)
        self.layer2 = nn.Linear(config.layer1_size, config.layer2_size)
        self.layer3 = nn.Linear(config.layer2_size, config.output_size)
        self.lr = config.lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        acc = (outputs.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        acc = (outputs.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
config = Config(input_size=784, layer1_size=128, layer2_size=64, output_size=10, lr=LR)
model = MNISTClassifier(config)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)

trainer = pl.Trainer(max_epochs=EPOCHS)
trainer.fit(model, trainloader, testloader)