
# setup training function
import torch
import torch_geometric
from tqdm.auto import tqdm

# setup accuracy metric
def accuracy(y_pred, y_true):
    return (y_pred==y_true).sum() / len(y_pred)

# train function
def train(model: torch.nn.Module,
          data: torch_geometric.data.data.Data,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Adam,
          epochs: int=10,
          print_results: bool=False,
          adjacency = None):

    # initilize training and validating results
    results = {
        'epoch'     : [],
        'train_loss': [],
        'train_acc' : [],
        'val_loss'  : [],
        'val_acc'   : []
    }
    
    # switch to training mode
    model.train()

    # loop over epochs
    for epoch in tqdm(range(epochs)):
        # zero our gradient
        optimizer.zero_grad()
        # compute logit
        if adjacency is not None:
            logit = model(data.x, adjacency)
        else:
            logit = model(data.x)
        # compute train loss
        train_loss = loss_fn(
            input=logit[data.train_mask],
            target=data.y[data.train_mask]
        )
        # train accuracy
        train_acc = accuracy(
            y_pred=torch.argmax(logit[data.train_mask], dim=1),
            y_true=data.y[data.train_mask]
        )
        # back propagation
        train_loss.backward()
        # optimizer step
        optimizer.step()
        # perform training validation every 20 steps
        if epoch % 20 == 0:
            # compute validation loss
            val_loss = loss_fn(
                input=logit[data.val_mask],
                target=data.y[data.val_mask]
            )
            # compute validation accuracy
            val_acc = accuracy(
                y_pred=torch.argmax(logit[data.val_mask], dim=1),
                y_true=data.y[data.val_mask]
            )

            # collecting results every 20 epochs
            results['epoch'].append(epoch)
            results['train_loss'].append(train_loss.item())
            results['train_acc'].append(train_acc.item())
            results['val_loss'].append(val_loss.item())
            results['val_acc'].append(val_acc.item())

            # printing training and validating result every 20 epochs
            if print_results:
                print(f"{epoch:5d} | {train_loss:.3f} | {train_acc:.3f} | {val_loss:.3f} | {val_acc:.3f}")

    # return train and validation results
    return results

def test(model: torch.nn.Module,
        data: torch_geometric.data.data.Data,
        adjacency=None):

    # switch model to evaluation mode
    model.eval()
    with torch.inference_mode():
        if adjacency is not None:
            test_logit = model(data.x, adjacency)
        else:
            test_logit = model(data.x)
            
        test_acc = accuracy(
            y_pred=torch.argmax(test_logit[data.test_mask], dim=1),
            y_true=data.y[data.test_mask]
        )
    # return test accuracy
    return test_acc


# train function for node regression
def train_regression(model: torch.nn.Module,
          data: torch_geometric.data.data.Data,
          epochs: int=10,
          print_results: bool=False):

    # initilize training and validating results
    results = {
        'epoch'     : [],
        'train_loss': [],
        'val_loss'  : [],
    }
    
    # switch to training mode
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-2, weight_decay=5e-4)
    
    # loop over epochs
    for epoch in tqdm(range(epochs)):
        
        # zero our gradient
        optimizer.zero_grad()
        
        # compute logit
        logit = model(data.x, data.edge_index)

        # compute train loss for regression
        train_loss = torch.nn.functional.mse_loss(
            logit.squeeze()[data.train_mask],
            data.y[data.train_mask].float()
        )

        # back propagation
        train_loss.backward()
        
        # optimizer step
        optimizer.step()
        
        # perform training validation every 20 steps
        if epoch % 20 == 0:
            # compute train loss for regression
            val_loss = torch.nn.functional.mse_loss(
                input=logit.squeeze()[data.val_mask],
                target=data.y[data.val_mask].float()
            )

            # collecting results every 20 epochs
            results['epoch'].append(epoch)
            results['train_loss'].append(train_loss.item())
            results['val_loss'].append(val_loss.item())

            # printing training and validating result every 20 epochs
            if print_results:
                print(f"{epoch:5d} | {train_loss:.3f} | {val_loss:.3f}")

    # return train and validation results
    return results

# test function for node regression task
def test_repression(model: torch.nn.Module,
                    data: torch_geometric.data.data.Data):

    # switch model to evaluation mode
    model.eval()
    with torch.inference_mode():
        test_logit = model(data.x, data.edge_index)

    return functional.mse_loss(out.squeeze()[data.test_mask], data.y[data.test_mask].float())
