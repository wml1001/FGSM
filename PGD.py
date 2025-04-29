import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pgd(model,image,label,epsilon,alpha,T):
    
    model,image,label = model.to(device),image.to(device),label.to(device)

    model.eval()

    adv_image = image.clone().detach()

    adv_image = adv_image + torch.zeros_like(adv_image).uniform_(-epsilon,epsilon)
    adv_image = torch.clamp(adv_image,0,1)

    for _ in range(T):

        adv_image.requires_grad_(True)

        output = model(adv_image)

        loss = nn.CrossEntropyLoss()(output,label)

        model.zero_grad()

        loss.backward()

        grad = adv_image.grad.data

        adv_image = adv_image + alpha * grad.sign()

        delta = torch.clamp(adv_image - image,-epsilon,epsilon)

        adv_image = torch.clamp(image + delta,0,1).detach()

        
    return adv_image