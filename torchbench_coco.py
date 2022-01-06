from torchbench.object_detection import COCO
from torchbench.utils import send_model_to_device
from torchbench.object_detection.transforms import Compose, ConvertCocoPolysToMask, ToTensor
import torchvision
import PIL

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
#maskrcnn_resnet50_fpn
model = torchvision.models.detection.__dict__[model_name](num_classes=91, pretrained=True)

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