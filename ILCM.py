import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ilcm(model, x, y, epsilon=0.3, alpha=0.03, T=10):
    model, x = model.to(device), x.to(device)
    model.eval()
    
    adv_image = x.clone().detach()
    
    # 初始化时动态获取最不可能类别（无梯度跟踪）
    with torch.no_grad():
        output = model(adv_image)
        y_fake = output.argmin(dim=-1)  # 固定初始目标标签
    
    for _ in range(T):
        adv_image.requires_grad_(True)
        
        if adv_image.grad is not None:
            adv_image.grad.zero_()
        
        output = model(adv_image)
        loss = nn.CrossEntropyLoss()(output, y_fake)  # 始终攻击初始最不可能类别
        
        model.zero_grad()
        loss.backward()
        
        grad = adv_image.grad.data
        adv_image = adv_image - alpha * grad.sign()
        
        # 限制扰动范围并裁剪到合法像素值
        delta = torch.clamp(adv_image - x, min=-epsilon, max=epsilon)
        adv_image = torch.clamp(x + delta, 0, 1).detach()
    
    return adv_image