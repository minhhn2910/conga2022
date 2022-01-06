import os
import tqdm 
import torch
import torchbench
from torch.utils.data import DataLoader
from torchbench.utils import send_model_to_device
from torchbench.object_detection.transforms import Compose, ConvertCocoPolysToMask, ToTensor
import torchvision
import PIL

from sotabencheval.object_detection import COCOEvaluator
from sotabencheval.utils import is_server

if is_server():
    DATA_ROOT = './.data/vision/coco'
else: # local settings
    DATA_ROOT = '/home/minh/CK-TOOLS/dataset-coco-2017-val/'

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

model = torchvision.models.detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=91, pretrained=True)

model, device = send_model_to_device(
    model, device='cuda', num_gpu=1
)
model.eval()

model_output_transform = coco_output_transform
send_data_to_device = coco_data_to_device
collate_fn = coco_collate_fn

test_dataset = torchbench.datasets.CocoDetection(
    root=os.path.join(DATA_ROOT, "val%s" % '2017'),
    annFile=os.path.join(
        DATA_ROOT, "annotations/instances_val%s.json" % '2017'
    ),
    transform=None,
    target_transform=None,
    transforms=transforms,
    download=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)
test_loader.no_classes = 91  # Number of classes for COCO Detection

iterator = tqdm.tqdm(test_loader, desc="Evaluation", mininterval=5)

evaluator = COCOEvaluator(
    root=DATA_ROOT,
    model_name='Mask R-CNN (ResNet-50-FPN)',
    paper_arxiv_id='1703.06870')

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

with torch.no_grad():
    for i, (input, target) in enumerate(iterator):
        input, target = send_data_to_device(input, target, device=device)
        original_output = model(input)
        output, target = model_output_transform(original_output, target)
        result = {
            tar["image_id"].item(): out for tar, out in zip(target, output)
        }
        result = prepare_for_coco_detection(result)

        evaluator.update(result)

        if evaluator.cache_exists:
            break

evaluator.save()