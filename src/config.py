import torch

DIMENSIONS = (1280, 960)
SELFIE_CLASS_NAME = "selfie"
BYSTANDER_CLASS_NAME = "bystander"
YOLO_MODEL_PATH = "models/yolo/best.pt"
SUPPORTED_MODELS = ["cnn", "vit", "vlm"]
SUPPORTED_MODES = ["blur", "bb"]
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
