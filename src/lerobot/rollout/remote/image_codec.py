import base64

import cv2
import numpy as np
import torch


def encode_image(img_bgr: np.ndarray, max_dim: int = 224, quality: int = 85) -> str:
    h, w = img_bgr.shape[:2]
    if h >= w:
        new_h = max_dim
        new_w = int(round(w * new_h / h))
    else:
        new_w = max_dim
        new_h = int(round(h * new_w / w))

    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    success, img_encoded = cv2.imencode(".jpg", img_resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(img_encoded.tobytes()).decode("ascii")


def decode_image_to_tensor(b64_str: str) -> torch.Tensor:
    raw_bytes = base64.b64decode(b64_str)
    img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return img_tensor.unsqueeze(0)
