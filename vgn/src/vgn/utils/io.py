import json
import yaml

import cv2
import numpy as np


def save_dict(data, path):
    """Serialize dict object to a JSON or YAML file."""
    suffix = path.suffix
    if suffix in [".json"]:
        _save_json(data, path)
    elif suffix in [".yaml"]:
        _save_yaml(data, path)
    else:
        raise ValueError("{} files are not supported.".format(suffix))


def load_dict(path):
    """Deserialize a dict object from a JSON or YAML file."""
    suffix = path.suffix
    if suffix in [".json"]:
        data = _load_json(path)
    elif suffix in [".yaml"]:
        data = _load_yaml(path)
    else:
        raise ValueError("{} files are not supported.".format(suffix))
    return data


def save_image(img, path):
    """Save image as a PNG file."""
    img = (1000.0 * img).astype(np.uint16)
    cv2.imwrite(str(path), img)


def load_image(path):
    """Load image from a PNG file."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) * 0.001
    return img


def _save_json(data, path):
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def _load_json(path):
    with path.open("r") as f:
        data = json.load(f)
    return data


def _save_yaml(data, path):
    with path.open("w") as f:
        yaml.dump(data, f)


def _load_yaml(path):
    with path.open("r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data
