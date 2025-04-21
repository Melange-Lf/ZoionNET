"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               metric_fn,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training metrics.
    In the form (train_loss, train_metric). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train metric values
    train_loss, train_metric = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate metric across all batches
        # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        y_pred_adjusted = torch.sigmoid(y_pred)
        train_metric += metric_fn(y_pred_adjusted, y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and metric per batch 
    train_loss = train_loss / len(dataloader)
    train_metric = train_metric / len(dataloader)
    return train_loss, train_metric

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              metric_fn,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing metrics.
    In the form (test_loss, test_metric). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test metric values
    test_loss, test_metric = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate metric
            # test_pred_labels = test_pred_logits.argmax(dim=1)
            y_pred_adjusted = torch.sigmoid(test_pred_logits)
            test_metric += (metric_fn(y_pred_adjusted, y).sum().item()/len(y_pred_adjusted))

    # Adjust metrics to get average loss and metric per batch 
    test_loss = test_loss / len(dataloader)
    test_metric = test_metric / len(dataloader)
    return test_loss, test_metric

# taken from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          metric_fn,
          epochs: int,
          device: torch.device,
          patience: int,
          min_delta: int) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing metric metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_metric: [...],
              test_loss: [...],
              test_metric: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_metric: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_metric: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_metric": [],
               "test_loss": [],
               "test_metric": []
    }
    
    # early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_metric = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          metric_fn=metric_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_metric = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          metric_fn=metric_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_metric: {train_metric:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_metric: {test_metric:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_metric"].append(train_metric)
        results["test_loss"].append(test_loss)
        results["test_metric"].append(test_metric)

        # if early_stopper.early_stop(test_loss):             
        #     print(f"early stopping at epoch {epoch}")
        #     break

    # Return the filled results at the end of the epochs
    return results
