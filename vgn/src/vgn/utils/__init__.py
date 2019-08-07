import cv2
import numpy as np


def save_image(fname, img):
    """Save image as a PNG file."""
    img = (1000. * img).astype(np.uint16)
    cv2.imwrite(fname, img)


def load_image(fname):
    """Load image from a PNG file."""
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) * 0.001
    return img
