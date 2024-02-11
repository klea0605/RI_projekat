import torch

def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for X, y in dataloader:
        # X = X.unsqueeze(1)
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = loss_fn(preds, y)

        # racunanje gradijenta
        loss.backward()
        # x_new = x - lr * grad
        optimizer.step()
        optimizer.zero_grad()

def cnn_train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for X, y in dataloader:
        X = X.unsqueeze(1)
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = loss_fn(preds, y)

        # racunanje gradijenta
        loss.backward()
        # x_new = x - lr * grad
        optimizer.step()
        optimizer.zero_grad()

def test_loop(dataloader, model, loss_fn, device):
    model.eval() # ne evaluira model, samo stavlja model u eval rezim - dropout i batchnorm nisu aktivni
    with torch.no_grad():
        num_same = 0
        for X, y in dataloader:
            # X = X.unsqueeze(1)
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            _, indices = torch.max(outputs, 1) # maks po dimenziji 1 jer je outputs.shape = (batch_size, num_classes)
            num_same += sum(indices == y).item()
        return num_same / len(dataloader.dataset)

def cnn_test_loop(dataloader, model, loss_fn, device):
    model.eval() # ne evaluira model, samo stavlja model u eval rezim - dropout i batchnorm nisu aktivni
    with torch.no_grad():
        num_same = 0
        for X, y in dataloader:
            X = X.unsqueeze(1)
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            _, indices = torch.max(outputs, 1) # maks po dimenziji 1 jer je outputs.shape = (batch_size, num_classes)
            num_same += sum(indices == y).item()
        return num_same / len(dataloader.dataset)