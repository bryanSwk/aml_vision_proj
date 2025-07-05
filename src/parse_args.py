import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intelligent Vision for Selective Anonymity"
    )
    parser.add_argument(
        "--source", type=str, default="0", help="Video Source or Path to Video File"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"data/output/output_{datetime.now()}.mp4",
        help="Path to Save Output Video",
    )
    parser.add_argument("--model", type=str, default="cnn", help="Model to Use")
    parser.add_argument(
        "--mode",
        type=str,
        default="blur",
        help="Type to Display: blur or bb (for bounding boxes)",
    )

    return parser.parse_args()
