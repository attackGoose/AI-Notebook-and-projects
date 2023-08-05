import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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

        y_logit = model(X)
        y_pred_label = torch.softmax(y_logit, dim=1).argmax(dim=1)

        #calculate the loss
        loss = loss_func(y_logit, y)
       
        optimizer.zero_grad()

        #the back propagation of the loss function works together with the gradient descent in optimizer.step()  to help the model learn/train
        loss.backward()

        optimizer.step()

        #accumulate the train loss and train acc per batch
        train_loss += loss 
        train_acc += ((y_pred_label == y).sum().item()/len(y_pred_label)) #original: (y_pred == y).sum().item()/len(y_pred)
        #if the y pred is = to y, we take the total amount of that through sum, then turn it into an item, and device it by the length of y_pred to get the accuracy

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)/len(data_loader.dataset)} samples")

    #adjust metrics to get the train loss and acc per batch
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
            test_logit = model(X) 
            test_pred = torch.softmax(test_logit, dim=1).argmax(dim=1) #the softmax isn't necessary but its there for completeness

            loss = loss_func(test_logit, y)

            test_loss += loss.item() #item just gets a single integer from whatever you call it on
            test_acc += ((test_pred == y).sum().item()/len(test_pred))

        #adjust metrics to get the test loss and acc per batch
        test_loss /= len(dataloader.dataset)
        test_acc /= len(dataloader.dataset)

        return test_loss, test_acc
    

#combines all the steps above
def epoch_loop_train(model: torch.nn.Module,
               train_dataloader: DataLoader, #torch.utils.data.DataLoader
               test_dataloader: DataLoader, 
               loss_func: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epochs: int = 5):
    """Trains the model"""

    #creating an empty results dictionary:
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step_multiclass(model=model,
                                                      data_loader=train_dataloader,
                                                      loss_func=loss_func,
                                                      optimizer=optimizer,
                                                      device=device)
        
        test_loss, test_acc = test_step_multiclass(model=model,
                                                   dataloader=test_dataloader,
                                                   loss_func=loss_func,
                                                   device=device)
        
        print(f"Epoch: {epoch} | Train Loss: {train_loss} | Train Acc: {train_acc} | Test Loss: {test_loss} | Test Acc: {test_acc}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results

