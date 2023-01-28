import torch
import torch.nn as nn

# take a transformer model and convert it to a combined model
class CombinedModel(nn.Module):
    def __init__(self, preprocess, transformer, thought_head, decision_head):
        super().__init__()
        self.preprocess = preprocess # patch embedding, class token, positional embedding
        self.transformer = transformer 
        self.thought_head = thought_head 
        self.decision_head = decision_head

    def forward(self, x, iters=0):
        """Preprocess, applies transformer + thought_head iters times, and then transformer one last time + decision head.
        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.preprocess(x)  # b,gh*gw+1,d 
        x = self.transformer(x) # b, gh*gw+1, d
        # every iteration, b, gh*gw+(i + 1), d -> b, gh*gw+(i + 2), d
        for i in range(iters):
            x = self.thought_head(x)
            x = self.transformer(x)
        x = self.decision_head(x[:, 0]) # b, d -> b, num_classes
        return x
