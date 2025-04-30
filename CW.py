import torch

class CW:
    def __init__(self, model, c=1e-1, k=0, lr=0.01, T=2000):  # 调整默认参数
        self.classifier = model
        self.c = c
        self.k = k
        self.lr = lr
        self.T = T

    def generate(self, x, target_class):
        x = x.clone().detach().clamp(0, 1)
        w = torch.arctanh(2 * (x - 0.5).clamp(min=-0.9999, max=0.9999))
        w = w.requires_grad_(True)
        
        optimizer = torch.optim.Adam([w], lr=self.lr)
        
        for _ in range(self.T):
            x_adv = (torch.tanh(w) + 1) / 2
            output = self.classifier(x_adv)
            
            if self.is_adv_image(output, target_class):
                break
                
            loss = self.loss(output, target_class, x, x_adv)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)  
            optimizer.step()
        
        return x_adv.detach().clamp(0, 1)

    def loss(self, output, target_class, x, x_adv):
        # L2距离（增加缩放因子）
        distance = torch.norm((x_adv - x).view(x.size(0), -1) / x.numel())
        
        # 分类损失方向
        target_logit = output[:, target_class]
        other_logits = output[:, torch.arange(output.size(1)) != target_class]
        max_other_logit = other_logits.max(dim=1).values
        
        # max( (max_Zi - Zt) , -k ) → 需要最小化这个值
        classification_loss = torch.clamp(target_logit - max_other_logit + self.k, max=0)
        
        return distance.mean() - self.c * classification_loss.mean()

    def is_adv_image(self, output, target_class):
        return output.argmax(dim=1) == target_class