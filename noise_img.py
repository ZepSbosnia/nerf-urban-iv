import os
import cv2
import numpy as np
from pathlib import Path
from typing import Literal
from tqdm import tqdm

def add_noise_rgb(rgb: np.ndarray, noise_type: Literal['gaussian', 's&p', 'poisson'] = 'gaussian') -> np.ndarray:
    if noise_type == 'gaussian':
        mean = 0
        stddev = 25
        noise = np.random.normal(mean, stddev, rgb.shape).astype(np.float32)
        noisy_img = np.clip(rgb.astype(np.float32) + noise, 0, 255)
        return noisy_img.astype(np.uint8)

    elif noise_type == 's&p':
        amount = 0.02
        noisy_img = rgb.copy()
        num_salt = np.ceil(amount * rgb.size * 0.5)
        coords = tuple([np.random.randint(0, i - 1, int(num_salt)) for i in rgb.shape[:2]])
        noisy_img[coords[0], coords[1], :] = 255
        num_pepper = np.ceil(amount * rgb.size * 0.5)
        coords = tuple([np.random.randint(0, i - 1, int(num_pepper)) for i in rgb.shape[:2]])
        noisy_img[coords[0], coords[1], :] = 0
        return noisy_img

    elif noise_type == 'poisson':
        vals = len(np.unique(rgb))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(rgb.astype(np.float32) * vals) / float(vals)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

def process_images(input_folder: str, output_folder: str, noise_type: str = 'gaussian'):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png'}

    for img_file in tqdm(list(input_path.iterdir()), desc="Adding noise"):
        if img_file.suffix.lower() in image_extensions:
            img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)  # Preserve alpha if present

            if img is None:
                print(f"Warning: Couldn't read image {img_file}")
                continue

            if img.shape[2] == 4:
                rgb = img[:, :, :3]
                alpha = img[:, :, 3:]
                noisy_rgb = add_noise_rgb(rgb, noise_type)
                noisy_img = np.concatenate([noisy_rgb, alpha], axis=-1)
            else:
                noisy_img = add_noise_rgb(img, noise_type)

            out_file = output_path / img_file.name
            cv2.imwrite(str(out_file), noisy_img)
# Example usage
if __name__ == "__main__":
    process_images(
        input_folder="/volume/torch-ngp_NODIR/VR_res/house/images",
        output_folder="/volume/torch-ngp_NODIR/VR_res/house_noise/images",
        noise_type="gaussian"  # Options: "gaussian", "s&p", "poisson"
    )
