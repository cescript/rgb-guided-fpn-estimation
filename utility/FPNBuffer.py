import os
import torch

# collects all fpn estimates and saves them as a single .pt file
class FPNBuffer:
    def __init__(self, path, subfolder):
        self.result_dir = os.path.join(path, subfolder)
        os.makedirs(self.result_dir, exist_ok=True)
        self.buffer = []

    def start(self):
        self.buffer = []

    # collect one window's FPN estimate, fpn: [2, H, W]
    def evaluate(self, fpn):
        self.buffer.append(fpn.detach().cpu())

    # save all collected fpn as a single tensor [N, 2, H, W]
    def save(self, name):
        if len(self.buffer) == 0:
            return
        fpn_stack = torch.stack(self.buffer, dim=0)
        path = os.path.join(self.result_dir, f"fpn_buffer_{name.lower()}.pt")
        torch.save(fpn_stack, path)
        print(f"[saved] {path} {list(fpn_stack.shape)}")