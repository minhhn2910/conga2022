from torchbench.object_detection import COCO
from torchbench.utils import send_model_to_device
from torchbench.object_detection.transforms import Compose, ConvertCocoPolysToMask, ToTensor
import torchvision
import PIL

import torch.nn as nn
import qtorch
from qtorch.quant import configurable_table_quantize, posit_quantize


def coco_data_to_device(input, target, device: str = "cuda", non_blocking: bool = True):
    input = list(inp.to(device=device, non_blocking=non_blocking) for inp in input)
    target = [{k: v.to(device=device, non_blocking=non_blocking) for k, v in t.items()} for t in target]
    return input, target

def coco_collate_fn(batch):
    return tuple(zip(*batch))

def coco_output_transform(output, target):
    output = [{k: v.to("cpu") for k, v in t.items()} for t in output]
    return output, target

transforms = Compose([ConvertCocoPolysToMask(), ToTensor()])

model_name = 'fasterrcnn_resnet50_fpn'
#model_name = 'maskrcnn_resnet50_fpn'
model = torchvision.models.detection.__dict__[model_name](num_classes=91, pretrained=True)


def other_weight(input):
    input = posit_quantize(input, nsize=16, es=1)
    return input

def other_activation(input):

    input = posit_quantize(input, nsize=16, es=1)
    return input

def linear_weight(input):
    input = posit_quantize(input, nsize=8, es=1, scale= 4.0)
    return input
def linear_activation(input):
    global act_data
    input = posit_quantize(input, nsize=8, es=1, scale= 0.5)
    return input


def forward_pre_hook_other(m, input):
    return (other_activation(input[0]),)

def forward_pre_hook_linear(m, input):

    return (linear_activation(input[0]),)


layer_count = 0
total_layer = 0

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) :
        module.weight.data = linear_weight(module.weight.data)
        module.register_forward_pre_hook(forward_pre_hook_linear)

        total_layer+=1
        layer_count +=1

    else: #should use fixedpoint or posit 16 for other layers 'weight
        if hasattr(module, 'weight'):
            total_layer +=1
            module.weight.data = other_weight(module.weight.data)
            module.register_forward_pre_hook(forward_pre_hook_other)

            #pass

print ("total %d  layers  ; using posit on %d conv/linear layers"%(total_layer, layer_count))


# Run the benchmark
COCO.benchmark(
    model=model,
    paper_model_name='Mask R-CNN (ResNet-50-FPN)',
    paper_arxiv_id='1703.06870',
    transforms=transforms,
    model_output_transform=coco_output_transform,
    send_data_to_device=coco_data_to_device,
    collate_fn=coco_collate_fn,
    batch_size=8,
    num_gpu=1
)