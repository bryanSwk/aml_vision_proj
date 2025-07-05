import cv2
from src.config import DIMENSIONS


def get_video_capture(source):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DIMENSIONS[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DIMENSIONS[1])

    return cap


def get_video_writer(path, cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))
