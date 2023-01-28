import torch
import torch.nn as nn

# take a transformer model and convert it to a combined model
class CombinedModel(nn.Module):
    def __init__(self, transformer, thought_head, decision_head):
        super().__init__()
        self.transformer = transformer
        self.thought_head = thought_head
        self.decision_head = decision_head

    def forward(self, x, iters=0):
        x = self.transformer(x)
        for i in range(iters):
            x = self.thought_head(x)
            x = self.transformer(x)
        x = self.decision_head(x)
        return x
