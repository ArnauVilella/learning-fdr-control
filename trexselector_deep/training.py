import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, evaluate=True, device="cpu"):
    """
    Train the model on the given data.
    
    Parameters
    ----------
    model : nn.Module
        The model to train
    train_loader : DataLoader
        DataLoader for training data
    test_loader : DataLoader
        DataLoader for testing data
    criterion : loss function
        The loss function to use for training
    optimizer : torch.optim.Optimizer
        The optimizer to use for training
    num_epochs : int, default=10
        Number of epochs to train for
    evaluate : bool, default=True
        Whether to evaluate the model after each epoch
    device : torch.device, default=device
        The device to use for computation
        
    Returns
    -------
    train_losses : list
        List of average training losses for each epoch
    test_losses : list
        List of average test losses for each epoch
    """
    model.train()
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", position=0)
        for _, data in progress_bar:
            Phis = data['phi'].to(device)
            betas = data['beta'].to(device)
            vs = data['v'].to(device)
            T_stops = data['T_stop'].to(device)
            Ls = data['L'].to(device)
            FDPs = data['fdp'].to(device)
            
            target = FDPs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(Phis, vs, T_stops, Ls)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Running Loss: {running_loss / len(train_loader):.6f}')
        if evaluate:
            losses_over_train, losses_over_test = evaluate_model(model, train_loader, test_loader, criterion)
            train_losses.append(sum(losses_over_train) / len(train_loader))
            test_losses.append(sum(losses_over_test) / len(test_loader))
        else:
            train_losses.append(-1)
            test_losses.append(-1)
    return train_losses, test_losses


def evaluate_model(model, train_loader, test_loader, criterion, verbose=True, device="cpu"):
    """
    Evaluate the model on training and test data.
    
    Parameters
    ----------
    model : nn.Module
        The model to evaluate
    train_loader : DataLoader
        DataLoader for training data
    test_loader : DataLoader
        DataLoader for testing data
    criterion : loss function
        The loss function to use for evaluation
    verbose : bool, default=True
        Whether to print evaluation results
    device : torch.device, default=device
        The device to use for computation
        
    Returns
    -------
    losses_over_train : list
        List of losses for each batch in the training set
    losses_over_test : list
        List of losses for each batch in the test set
    """
    model.eval()
    losses_over_train = []
    with torch.no_grad():
        for data in train_loader:
            Phis = data['phi'].to(device)
            vs = data['v'].to(device)
            T_stops = data['T_stop'].to(device)
            Ls = data['L'].to(device)
            FDPs = data['fdp'].to(device)
            
            target = FDPs.unsqueeze(1)
            outputs = model(Phis, vs, T_stops, Ls)
            loss = criterion(outputs, target)
            losses_over_train.append(loss.item())
    if verbose:
        print(f'Train Loss: {sum(losses_over_train) / len(train_loader):.6f}')
    
    losses_over_test = []
    with torch.no_grad():
        for data in test_loader:
            Phis = data['phi'].to(device)
            vs = data['v'].to(device)
            T_stops = data['T_stop'].to(device)
            Ls = data['L'].to(device)
            FDPs = data['fdp'].to(device)
            
            target = FDPs.unsqueeze(1)
            outputs = model(Phis, vs, T_stops, Ls)
            loss = criterion(outputs, target)
            losses_over_test.append(loss.item())
    if verbose:
        print(f'Test Loss: {sum(losses_over_test) / len(test_loader):.6f}')
    
    return losses_over_train, losses_over_test


def get_loader_and_infer_FDP(dataloader, model, device="cpu"):
    """
    Get the actual and predicted FDP values from the dataloader.
    
    Parameters
    ----------
    dataloader : DataLoader
        DataLoader containing data to evaluate
    model : nn.Module
        The trained model to use for prediction
    device : torch.device, default=device
        The device to use for computation
        
    Returns
    -------
    loader_FDP : list
        List of actual FDP values
    infer_FDP : list
        List of predicted FDP values
    """
    model.eval()

    loader_FDP_tensors = []
    infer_FDP_tensors = []
    with torch.no_grad():
        for data in dataloader:
            Phis = data['phi'].to(device)
            vs = data['v'].to(device)
            T_stops = data['T_stop'].to(device)
            Ls = data['L'].to(device)
            FDPs = data['fdp'].to(device)
            
            loader_FDP_tensors.append(FDPs)
            infer_FDP_tensors.append(model(Phis, vs, T_stops, Ls).detach())

    loader_FDP = torch.cat(loader_FDP_tensors).cpu().numpy().squeeze().tolist()
    infer_FDP = torch.cat(infer_FDP_tensors).cpu().numpy().squeeze().tolist()
    
    return loader_FDP, infer_FDP
