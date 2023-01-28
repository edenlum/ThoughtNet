import torch
import torch.nn as nn
from copy import deepcopy, copy
from torch.utils.data import DataLoader


def train(model, epochs, train_loader, test_loader, opt=None, loss=None, test=True, loss_stop=0.0, save=False, name="checkpoints/model.pt"):
    if loss == None:
        loss = torch.nn.CrossEntropyLoss()
    if opt == None:
        opt = torch.optim.Adam(model.parameters(), lr=1e-7)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
    model.cuda()
    lrs = []
    losses = []
    accuracies = []
    for epoch in range(epochs):
        model.train()
        for idx, batch in enumerate(train_loader):
            opt.zero_grad()
            x, y = batch[0].cuda(), batch[1].cuda()
            if type(loss) == torch.nn.CrossEntropyLoss:
                y = nn.functional.one_hot(y, num_classes=10).float()
            pred_y = model(x)
            l = loss(pred_y, y)
            losses.append(l.item())
            lrs.append(opt.param_groups[0]['lr'])
            l.backward()
            opt.step()
            scheduler.step()

        print(f"loss after epoch {epoch}: {l.item()}")
        if test:
            assert test_loader != None
            test_model(model, test_loader)
        if l.item() < loss_stop:
            break
    
    if save:
        torch.save(model, name)

    return (lrs, losses, accuracies)


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

def add_norm_output_to_dataset(dataset):
        def hook(model, input, output):
            dataset.append([copy(output)])
        return hook

def get_layer(model, layer_name):
    layer = [child for child in model.named_children() if child[0] == layer_name][0][1]
    return layer

def create_overfit_dataset(model, train_dataset, layer_name="norm", size=None):
    opt_class_token = torch.optim.Adam([{
        'params': model.get_parameter('class_token')
    }], lr = 0.1)
    model.train()
    layer = get_layer(model, layer_name)
    small_train_dataset = list(train_dataset)[:size]
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

def test_thought(model, dataset, train_dataset, thought_model, layer_name="norm"):
    num_correct, num_correct_orig, num_correct_thought, worse, better, total, thought_better, thought_worse = 0, 0, 0, 0, 0, 0, 0, 0
    model.eval()
    thought_model.eval()
    
    # saving base class_token
    base_class_token = deepcopy(model.get_parameter('class_token'))
    # layer = get_layer(model, layer_name)
    for (x, y), (a, a_star) in zip(list(train_dataset)[:len(dataset)-1], dataset[:-1]):
        # original
        state_dict = model.state_dict()
        state_dict["class_token"] = base_class_token
        model.load_state_dict(state_dict)
        # a_list = []
        # handle = layer.register_forward_hook(add_norm_output_to_dataset(a_list))
        pred_y_orig = model(x.unsqueeze(0).cuda()).argmax(dim=1)
        # a = a_list[0]
        # handle.remove()

        # with optimized token
        state_dict = model.state_dict()
        state_dict["class_token"] = a_star
        model.load_state_dict(state_dict)
        pred_y = model(x.unsqueeze(0).cuda()).argmax(dim=1)
        
        # with predicted optimized token
        pred_a_star = thought_model(a)
        state_dict = model.state_dict()
        state_dict["class_token"] = pred_a_star[:, 0:1]
        model.load_state_dict(state_dict)
        pred_y_thought = model(x.unsqueeze(0).cuda()).argmax(dim=1)
        
        if torch.sum(y == pred_y) > torch.sum(y == pred_y_orig):
            better += 1
        if torch.sum(y == pred_y) < torch.sum(y == pred_y_orig):
            worse += 1
        if torch.sum(y == pred_y_thought) > torch.sum(y == pred_y_orig):
            thought_better += 1
        if torch.sum(y == pred_y_thought) < torch.sum(y == pred_y_orig):
            thought_worse += 1
        num_correct += torch.sum(y == pred_y)
        num_correct_orig += torch.sum(y == pred_y_orig)
        num_correct_thought += torch.sum(y == pred_y_thought)
        total += 1 # batch size
    print(f"Accuracy: {num_correct/total}")
    print(f"Accuracy original: {num_correct_orig/total}")
    print(f"Accuracy thought: {num_correct_thought/total}")
    print(f"Worse: {worse}")
    print(f"Better: {better}")
    print(f"Thought worse: {thought_worse}")
    print(f"Thought better: {thought_better}")