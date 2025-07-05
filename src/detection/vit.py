import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import src.config as config
from src.processing.censors import apply_blur, draw_bb

model_path = r"F:\amlp2\models\checkpoint-624"

processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
model = AutoModelForObjectDetection.from_pretrained(model_path).to(config.DEVICE)


def process_vit(frame, mode):
    inputs = processor(images=frame, return_tensors="pt").to(config.DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([frame.shape[:2]]).to(config.DEVICE)
    results = processor.post_process_object_detection(
        outputs, threshold=0.3, target_sizes=target_sizes
    )[0]

    for label, box in zip(results["labels"], results["boxes"]):
        if mode == "blur":
            if cls_id:
                frame = apply_blur(frame, map(int, box))
        elif mode == "bb":
            if label:
                frame = draw_bb(
                    frame, map(int, box), config.BYSTANDER_CLASS_NAME, (0, 0, 255)
                )
            else:
                frame = draw_bb(
                    frame, map(int, box), config.SELFIE_CLASS_NAME, (255, 0, 0)
                )

    return frame
