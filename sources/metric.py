
def accuracy(y_pred, y_true):
    return torch.sum(y_pred == y_true) / len(y_true)

