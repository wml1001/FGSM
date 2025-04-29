import torch
import torch.nn as nn

def bim(model,image,label,epsilon=0.3,alpha=0.03,T=10):
    
    adv_images = image.clone().detach().requires_grad_(True)

    model.eval()

    for _ in range(T):

        model.zero_grad() 

        if adv_images.grad is not None:
            adv_images.zero_grad()

        output = model(adv_images)

        loss = nn.CrossEntropyLoss()(output,label)
        
        loss.backward()

        grad = adv_images.grad.data

        with torch.no_grad():
            adv_images = adv_images + alpha * grad.sign()
            
        # 直接裁剪到 [x-ε, x+ε] 和像素值范围
        adv_images = torch.clamp(adv_images, image-epsilon, image+epsilon).clamp(0,1)
        adv_images = adv_images.requires_grad_(True)

    
    return adv_images.detach()
