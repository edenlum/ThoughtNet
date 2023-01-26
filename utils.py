import torch
import torch.nn as nn
from copy import deepcopy, copy
from torch.utils.data import DataLoader


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
            assert test_loader != None
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

def create_overfit_dataset(model, train_dataset):
    opt_class_token = torch.optim.Adam([{
        'params': model.get_parameter('class_token')
    }], lr = 0.1)
    model.train()
    layer_name = "norm"
    layer = [child for child in model.named_children() if child[0] == layer_name][0][1]
    def add_norm_output_to_dataset(dataset):
        def hook(model, input, output):
            dataset.append([copy(output)])
        return hook


    small_train_dataset = list(train_dataset)[:1000]
    dataset = []
    # saving base class_token
    base_class_token = deepcopy(model.get_parameter('class_token'))
    for s in small_train_dataset:
        # 1 sample
        train_loader_batch1 = DataLoader([s], batch_size=1, shuffle=True)
        # adding a to dataset
        handle = layer.register_forward_hook(add_norm_output_to_dataset(dataset))
        model(s[0].unsqueeze(0).cuda())
        handle.remove()
        # overfitting
        train(model, 100, train_loader_batch1, test_loader=None, opt = opt_class_token, test=False, loss_stop = 0.01)
        # adding a* to dataset
        dataset[-1].append(deepcopy(model.get_parameter('class_token')))
        # loading back base class_token
        state_dict = model.state_dict()
        state_dict["class_token"] = base_class_token
        model.load_state_dict(state_dict)
    return dataset

def sanity_check(model, dataset, train_dataset):
    num_correct, num_correct_orig, worse, better, total = 0, 0, 0, 0, 0
    model.eval()
    base_class_token = deepcopy(model.get_parameter('class_token'))
    for (x, y), (a, a_star) in zip(train_dataset[:len(dataset)-1], dataset[:-1]):
        # original
        state_dict = model.state_dict()
        state_dict["class_token"] = base_class_token
        model.load_state_dict(state_dict)
        pred_y_orig = model(x.unsqueeze(0).cuda())
        pred_y_orig = pred_y_orig.argmax(dim=1)

        # with optimized token
        state_dict = model.state_dict()
        state_dict["class_token"] = a_star
        model.load_state_dict(state_dict)
        pred_y = model(x.unsqueeze(0).cuda())
        pred_y = pred_y.argmax(dim=1)
        if torch.sum(y == pred_y) > torch.sum(y == pred_y_orig):
            better += 1
        if torch.sum(y == pred_y) < torch.sum(y == pred_y_orig):
            worse += 1
        num_correct += torch.sum(y == pred_y)
        num_correct_orig += torch.sum(y == pred_y_orig)
        total += 1 # batch size
    print(f"Accuracy: {num_correct/total}")
    print(f"Accuracy original: {num_correct_orig/total}")
    print(f"Worse: {worse}")
    print(f"Better: {better}")