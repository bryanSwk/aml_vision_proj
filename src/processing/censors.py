import cv2


def apply_blur(img, box, ksize=100):
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img

    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)

    blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    img[y1:y2, x1:x2] = blurred
    return img


def apply_mosaic(img, box, scale=0.1):
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img
    small = cv2.resize(roi, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = mosaic
    return img


def draw_bb(img, box, label, color):
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if label:
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        text_width, text_height = text_size

        cv2.rectangle(
            img, (x1, y1 - text_height - 4), (x1 + text_width + 2, y1), color, -1
        )
        cv2.putText(
            img,
            label,
            (x1 + 1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return img
