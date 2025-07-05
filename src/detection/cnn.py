from ultralytics import YOLO
import src.config as config
from src.processing.censors import apply_blur, draw_bb

model = YOLO(config.MODEL_PATH)
names = model.names


def process_cnn(frame, mode):
    results = model(frame, agnostic_nms=True)[0]
    boxes = results.boxes

    for i in range(len(boxes.cls)):
        cls_id = int(boxes.cls[i])
        cls_name = names[cls_id]
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()

        if mode == "blur":
            if cls_name == config.BYSTANDER_CLASS_NAME:
                frame = apply_blur(frame, map(int, xyxy))
        elif mode == "bb":
            if cls_name == config.BYSTANDER_CLASS_NAME:
                frame = draw_bb(frame, map(int, xyxy), "bystander", (0, 0, 255))
            elif cls_name == config.SELFIE_CLASS_NAME:
                frame = draw_bb(frame, map(int, xyxy), "selfie", (255, 0, 0))

    return frame
