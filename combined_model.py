import torch
import torch.nn as nn

# take a transformer model and convert it to a combined model
class CombinedModel(nn.Module):
    def __init__(self, preprocess, transformer, thought_head, decision_head, iters = 0):
        super().__init__()
        self.preprocess = preprocess # patch embedding, class token, positional embedding
        self.transformer = transformer 
        self.thought_head = thought_head 
        self.decision_head = decision_head
        self.iters = iters

    def forward(self, x):
        """Preprocess, applies transformer + thought_head iters times, and then transformer one last time + decision head.
        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.preprocess(x)  # b,gh*gw+1,d 
        x = self.transformer(x) # b, gh*gw+1, d
        # every iteration, b, gh*gw+(i + 1), d -> b, gh*gw+(i + 2), d
        for i in range(self.iters):
            x = self.thought_head(x)
            x = self.transformer(x)
        x = self.decision_head(x) # b, d -> b, num_classes
        return x

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.thought_head.parameters(), lr=lr)


class ViTPreprocess(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.patch_embedding = vit.patch_embedding
        self.class_token = vit.class_token
        self.positional_embedding = vit.positional_embedding

    def forward(self, x):
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw+1,d 
        return x

class ViTDecisionHead(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.norm = vit.norm
        self.fc = vit.fc

    def forward(self, x):
        x = self.norm(x)[:, 0]  # b,d
        x = self.fc(x)  # b,num_classes
        return x

