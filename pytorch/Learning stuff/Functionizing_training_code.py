import torch
from torch.utils.data import DataLoader

# functionizing the training step
def train_step_multiclass(model: torch.nn.Module,
               data_loader: DataLoader, #torch.utils.data.DataLoader
               loss_func: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    
    """Trains the model using train_dataloader"""

    train_loss, train_acc = 0, 0
    
    #train mode
    model.train()

    #add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):
        #sending data to target device
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        #calculate the loss
        loss = loss_func(y_pred, y)
       
        optimizer.zero_grad()

        #the back propagation of the loss function works together with the gradient descent in optimizer.step()  to help the model learn/train
        loss.backward()

        optimizer.step()

        #accumulate the train loss and train acc per batch
        train_loss += loss 
        train_acc += (y_pred == y).sum()/len(y_pred) #original: (y_pred == y).sum().item()/len(y_pred)
        #if the y pred is = to y, we take the total amount of that through sum, then turn it into an item, and device it by the length of y_pred to get the accuracy

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)/len(data_loader.dataset)} samples")

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return f"train loss: {train_loss:5f} | Train acc: {train_acc:2f}"

def test_step_multiclass(model: torch.nn.Module,
                         dataloader: DataLoader,
                         loss_func: torch.nn.Module,
                         device: torch.device):
    """Tests the model"""

    test_loss, test_acc = 0, 0

    #putting the model in eval mode
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            #forward pass
            test_pred = model(X)

            loss = loss_func(test_pred, y)

            test_loss += loss
            test_acc += (test_pred == y).sum().item()/len(test_pred)

        test_loss /= len(dataloader.dataset)
        test_acc /= len(dataloader.dataset)