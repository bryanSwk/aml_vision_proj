import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import cv2
from src.processing.censors import apply_blur, draw_bb
import src.config as config

model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(config.DEVICE)

MAIN_CHAR_LABELS = [
    ["main character", "vlogger", "selfie taker"],
    [
        "main character of the photo",
        "person who is vlogging",
        "person taking the selfie",
        "person in the foreground",
    ],
]
FACE_LABELS = [["face"]]


def get_bboxes(frame, text_labels):
    res = []

    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for label in text_labels:
        inputs = processor(images=frame, text=label, return_tensors="pt").to(
            config.DEVICE
        )
        with torch.no_grad():
            outputs = model(**inputs)
        temp_res = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[frame.size[::-1]],
        )
        if len(temp_res[0]["scores"]) > 0:
            res.append(temp_res[0])
    if res:
        processed_res = {
            "scores": torch.cat([r["scores"] for r in res], dim=0),
            "boxes": torch.cat([r["boxes"] for r in res], dim=0),
            "text_labels": sum([r["text_labels"] for r in res], []),
            "labels": sum([r["labels"] for r in res], []),
        }
    else:
        processed_res = None

    return processed_res


def calc_overlap(face_box, main_char_box):
    x1 = max(face_box[0], main_char_box[0])
    y1 = max(face_box[1], main_char_box[1])
    x2 = min(face_box[2], main_char_box[2])
    y2 = min(face_box[3], main_char_box[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])

    return (intersection_area / face_area) if face_area > 0 else 0


def find_main_face(main_char_res, face_res, threshold=1):
    idx = torch.argmax(main_char_res["scores"]).item() if main_char_res else -1
    if idx == -1:
        face_res["main_face_overlap"] = []
        face_res["main_face_labels"] = [False for i in range(len(face_res["boxes"]))]
        return face_res

    main_char_bbox = main_char_res["boxes"][idx].cpu().tolist()
    face_bboxes = face_res["boxes"].tolist()
    res = face_res
    face_res["main_face_labels"] = []
    face_res["main_face_overlap"] = []

    for id, face_bbox in enumerate(face_bboxes):
        overlap = calc_overlap(face_bbox, main_char_bbox)
        res["main_face_overlap"].append(overlap)
        res["main_face_labels"].append(overlap >= threshold)

    return res


def process_vlm(frame, mode):
    main_char_res = get_bboxes(frame, MAIN_CHAR_LABELS)
    face_res = get_bboxes(frame, FACE_LABELS)
    res = find_main_face(main_char_res, face_res)

    for label, box in zip(res["main_face_labels"], res["boxes"]):
        if mode == "blur":
            if not label:
                frame = apply_blur(frame, map(int, box))
        elif mode == "bb":
            if not label:
                frame = draw_bb(
                    frame, map(int, box), config.BYSTANDER_CLASS_NAME, (0, 0, 255)
                )
            else:
                frame = draw_bb(
                    frame, map(int, box), config.SELFIE_CLASS_NAME, (255, 0, 0)
                )

    return frame
