import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fgsm(model,x,y,epsilon):

    model,x,y = model.to(device),x.to(device),y.to(device)

    model.eval()

    x.requires_grad = True

    output = model(x)

    loss = nn.CrossEntropyLoss()(output,y)

    model.zero_grad()
    loss.backward()


    data_grad = x.grad.data

    perturbed_x = x + epsilon * data_grad.sign()

    perturbed_x = torch.clamp(perturbed_x,0,1)

    return perturbed_x
