import os
import cv2
import src.config as config
from src.parse_args import parse_args
from src.processing.video_io import get_video_capture, get_video_writer


def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = get_video_capture(source)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video source: {args.source}")
    if args.model not in config.SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model format: {args.model}")
    if args.mode not in config.SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode format: {args.mode}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = get_video_writer(args.output, cap)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.model == "cnn":
            try:
                from src.detection.cnn import process_cnn

                processed_frame = process_cnn(frame, args.mode)
            except Exception as e:
                print(f"Error running YOLO inference: {e}")

        elif args.model == "vlm":
            try:
                from src.detection.vlm import process_vlm

                processed_frame = process_vlm(frame, args.mode)
            except Exception as e:
                print(f"Error running VLM inference: {e}")

        elif args.model == "vit":
            try:
                from src.detection.vit import process_vit

                processed_frame = process_vit(frame, args.mode)
            except Exception as e:
                print(f"Error running RT-DETR inference: {e}")

        writer.write(processed_frame)
        cv2.imshow("IVSA", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {args.output}")


if __name__ == "__main__":
    main()
