import torch 
from torch import nn, Tensor
from typing import Union, Tuple
import torchvision
from torchvision.transforms import v2 
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import random
import os
from model import Model 


seed_number = 42
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
random.seed(seed_number)
torch.manual_seed(seed_number)
torch.cuda.manual_seed(seed_number)
np.random.seed(seed_number)
os.environ['PYTHONHASHSEED'] = str(seed_number)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64 


def downloadData(batch_size, download=True):


    transforms = v2.Compose(
        [
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(1),
            v2.Normalize((0.5, ), (0.5, ))
        ]
    )

    trainset = torchvision.datasets.MNIST('./', train=True, transform=transforms, download=download)
    train_subset, val_subset = torch.utils.data.random_split(
        trainset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
    )

    testset = torchvision.datasets.MNIST('./', train=False, transform=transforms, download=download)
        
    train_data = torch.utils.data.DataLoader(train_subset,  batch_size=batch_size, shuffle=True, num_workers=2)
    test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    val_data = torch.utils.data.DataLoader(val_subset,  batch_size=batch_size, shuffle=True, num_workers=2)
    return train_data, val_data, test_data

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def calculateAccuracy(predicted, targets):
    predicted = nn.functional.softmax(predicted, dim=0)    # print(predicted[0], targets[0])
    pred_no = torch.argmax(predicted, dim=1)
    # print(predicted)
    right = torch.sum(torch.eq(pred_no, targets).int())
    return right / len(pred_no)

def validateModel(model, testIter, loss, device):
    loss_per_batch = []
    acc_per_batch = []
    with torch.no_grad():
        model.eval()
        for _, (X, y) in enumerate(testIter):
            X, y = X.to(device), y.to(device)
            out = model(X)
            l = loss(out, y)
            a = calculateAccuracy(out, y)
            loss_per_batch.append(l.item())
            acc_per_batch.append(a.item())
        mean_acc = sum(acc_per_batch) / len(acc_per_batch)
        meanloss = sum(loss_per_batch)/len(loss_per_batch)
    return loss_per_batch, meanloss, acc_per_batch, mean_acc


def train(trainIter, testIter, model,
          device=device,
          epochs=100, 
          optim=None, 
          loss=None,
          scheduler=None
          ):
    logs_dic = {
        "valildationLoss": [],
        "trainingLoss" : [],
        "validationAccuracy": [],
        "trainingAccuracy": []
    }
    for epoch in range(epochs):
        train_loss_per_batch = []
        train_acc_per_batch = []
        with tqdm(trainIter, unit="batches") as tepoch:
            for _, (X, y) in enumerate(tepoch):
                model.train()
                optim.zero_grad()
                X, y = X.to(device), y.to(device)
                out = model(X)
                l = loss(out, y)
                acc = calculateAccuracy(out, y)
                train_acc_per_batch.append(acc.item())
                train_loss_per_batch.append(l.item())
                tepoch.set_description(f"Epoch {epoch + 1}")
                tepoch.set_postfix(loss=l.item(), accuracy=acc.item())
                l.backward()
                optim.step()
        val_loss, mean_val_loss, val_acc, mean_val_acc = validateModel(model, testIter, loss=loss, device=device)
        print(f"The validation loss is: {mean_val_loss}")
        print(f"The validation accuracy is: {mean_val_acc}")
        logs_dic['valildationLoss'].append(val_loss)
        logs_dic['trainingLoss'].append(train_loss_per_batch)
        logs_dic['trainingAccuracy'].append(train_acc_per_batch)
        logs_dic['validationAccuracy'].append(val_acc)
        if scheduler: 
            if scheduler.__module__ == lr_scheduler.__name__:
                scheduler.step()
            else:
                for param_group in optim.param_groups:
                    lr = scheduler(epoch)
                    param_group['lr'] = lr

    return logs_dic

def test(testIter, model, device=device):
    with torch.no_grad():
        model.eval()
        acc_batch = []
        for _, (X, y) in enumerate(tqdm(testIter)):
            X, y = X.to(device), y.to(device)
            out = model(X)
            a = calculateAccuracy(out, y)
            acc_batch.append(a.item())
    total_acc = sum(acc_batch) / len(acc_batch)
    return total_acc


if __name__ == "__main__":
    trainloader, valloader, testloader = downloadData(BATCH_SIZE, download=False)
    model = Model()
    # optim = optim.Adam(model.parameters(), lr=0.0001 )
    optim = optim.SGD(model.parameters(), lr=0.009, momentum=0.93, dampening=0.05, weight_decay=0.009)
    # optim = optim.SGD(model.parameters(), lr=0.0023)
    acc = test(testloader, model)
    loss = nn.CrossEntropyLoss()
    print("The test accuracy is: ", acc)
    logs = train(trainloader, valloader, model, 
                 device=device, epochs=20, optim=optim, loss=loss)
    acc = test(testloader, model)
    print("The test accuracy is: ", acc)

