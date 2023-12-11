import torch_geometric
import torch
from tqdm.auto import tqdm
from sources.metrics import accuracy

# setup training step function
def train_step(model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               data: torch_geometric.data.data.Data,
              adjacency: torch.Tensor):

    # switch to training mode
    model.train()
    
    # forward pass
    if adjacency is not None:
        out = model(data.x, adjacency)
    else:
        out = model(data.x)
    
    # training loss
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    
    # training accuracy
    train_acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
    
    # zero out gradient
    optimizer.zero_grad()
    
    # back propagation
    loss.backward()
    
    # optimizer step
    optimizer.step()

    return loss.item(), train_acc
    

# setup test step function
def test_step(model: torch.nn.Module,
              data: torch_geometric.data.data.Data,
              adjacency: torch.Tensor,
             loss_fn: torch.nn.Module):
    
    # switch to evaluation mode
    model.eval()
    # turn on torch inference mode
    with torch.inference_mode():
        # test forward pass
        if adjacency is not None:
            out = model(data.x, adjacency)
        else:
            out = model(data.x)
        # test loss
        test_loss = loss_fn(out[data.test_mask], data.y[data.test_mask])
        
        # test accuracy
        test_acc = accuracy(out[data.test_mask].argmax(dim=1), data.y[data.test_mask]).item()
        
    # test accuracy
    return test_loss, test_acc

# setup train function
def train(model: torch.nn.Module,
         loss_fn: torch.nn.Module,
         optimizer: torch.optim.Optimizer,
         data: torch_geometric.data.data.Data,
          adjacency: torch.Tensor,
         epochs: int=100):

    results = {
        'train_loss': [],
        'train_acc' : [],
        'test_loss' : [],
        'test_acc'  : []
    }
    
    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            data=data,
            adjacency=adjacency
        )

        test_loss, test_acc = test_step(
            model=model,
            loss_fn=loss_fn,
            data=data,
            adjacency=adjacency
        )

        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        
    return results
