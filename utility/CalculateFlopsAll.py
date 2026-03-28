import torch
import models
from utility.GetDevice import GetDevice
from torch.utils.flop_counter import FlopCounterMode

# set the model names to be evaluated
model_names = ["SAFTA-RGB", "DCGAN", "MULTIVIEW", "D1WLS", "DLSNUC", "EMPTY"]

# get flops for each model
K = 12
device = GetDevice()
inp_rgb = [torch.randn((3, 288, 288), device=device) for _ in range(K)]
inp_irn = [torch.randn((1, 288, 288), device=device) for _ in range(K)]

# loop over all models and test the results
for mid, model_name in enumerate(model_names):
    # create a model given opt.model and other options
    model = models.GenerateModel(model_name)

    # after model creation
    num_params = 0
    for attr_name in dir(model):
        attr = getattr(model, attr_name)
        if isinstance(attr, torch.nn.Module):
            num_params += sum(p.numel() for p in attr.parameters())
            # print(f"  {attr_name}: {num_params / 1e3:.1f}K params")

    # measure the flops
    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        model.forward(inp_irn, inp_rgb)

    total_flops = flop_counter.get_total_flops()
    print("%10s \t Params: %.2fK \t FLOPs: %.2f GFlops" % (model_name, num_params / 1e3, total_flops / (K * 1e9)))
