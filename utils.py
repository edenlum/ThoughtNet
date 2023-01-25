import torch
import torch.nn as nn


def train(model, epochs, train_loader, test_loader, opt=None, loss=None, test=True, loss_stop=0.0, save=False, name="checkpoints/model.pt"):
    if loss == None:
        loss = torch.nn.CrossEntropyLoss()
    if opt == None:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.cuda()
    for epoch in range(epochs):
        model.train()
        for idx, batch in enumerate(train_loader):
            opt.zero_grad()
            x, y = batch[0].cuda(), batch[1].cuda()
            if type(loss) == torch.nn.CrossEntropyLoss:
                y = nn.functional.one_hot(y, num_classes=10).float()
            pred_y = model(x)
            l = loss(pred_y, y)
            l.backward()
            opt.step()

        print(f"loss after epoch {epoch}: {l.item()}")
        if test:
            test_model(model, test_loader)
        if l.item() < loss_stop:
            break
    
    if save:
        torch.save(model, name)


def test_model(model, test_loader):
    model.eval()
    num_correct = 0
    total = 0
    for idx, batch in enumerate(test_loader):
        x, y = batch[0].cuda(), batch[1].cuda()
        pred_y = model(x)
        pred_y = pred_y.argmax(dim=1)
        num_correct += torch.sum(y == pred_y)
        total += y.shape[0]  # batch size
    print(f"Accuracy: {num_correct/total}")
    return num_correct/total
