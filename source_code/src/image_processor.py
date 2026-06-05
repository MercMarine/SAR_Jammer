"""

Author - MercMarine
GitHub - https://github.com/MercMarine

image_processor.py - Классы для работы с изображениями и наложения помех

"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ImageLoader:
    @staticmethod
    def load_image_original_size(image_path):
        img = Image.open(image_path)
        img = img.convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array

# Отладочные методы

class InterferenceApplier:
    @staticmethod
    def apply_azimuth_interference(image, interference_signal, center_range=250,
                                   width=50, strength=0.3):
        img_with_interference = image.copy()
        img_height, img_width = image.shape

        if np.max(np.abs(interference_signal)) > 0:
            interference_signal = interference_signal / np.max(np.abs(interference_signal))

        start_range = max(0, center_range - width // 2)
        end_range = min(img_height, center_range + width // 2)
        actual_width = end_range - start_range

        if actual_width <= 0:
            return img_with_interference

        signal_len = len(interference_signal)
        if signal_len != img_width:
            if signal_len < img_width:
                repeats = int(np.ceil(img_width / signal_len))
                interference_signal = np.tile(interference_signal, repeats)[:img_width]
            else:
                interference_signal = interference_signal[:img_width]

        interference_2d = np.outer(np.ones(actual_width), interference_signal)
        img_with_interference[start_range:end_range, :] += strength * interference_2d
        img_with_interference = np.clip(img_with_interference, 0, 1)
        return img_with_interference

    @staticmethod
    def apply_range_interference(image, interference_signal, center_azimuth=250,
                                 width=50, strength=0.3, use_sinc_envelope=True):
        img_with_interference = image.copy()
        img_height, img_width = image.shape  # [дальность, азимут]

        # Нормализуем сигнал помехи
        if np.max(np.abs(interference_signal)) > 0:
            interference_signal = interference_signal / np.max(np.abs(interference_signal))


        start_azimuth = max(0, center_azimuth - width // 2)
        end_azimuth = min(img_width, center_azimuth + width // 2)
        actual_width = end_azimuth - start_azimuth

        if actual_width <= 0:
            return img_with_interference

        signal_len = len(interference_signal)
        if signal_len != img_height:
            if signal_len < img_height:
                repeats = int(np.ceil(img_height / signal_len))
                interference_signal = np.tile(interference_signal, repeats)[:img_height]
            else:
                interference_signal = interference_signal[:img_height]

        if use_sinc_envelope:
            azimuth_coords = np.linspace(-1.0, 1.0, actual_width)
            sinc_env = np.sinc(azimuth_coords)
            sinc_env = np.clip(sinc_env, 0, None)
            interference_2d = np.outer(interference_signal, sinc_env)
        else:
            interference_2d = np.outer(interference_signal, np.ones(actual_width))

        img_with_interference[:, start_azimuth:end_azimuth] += strength * interference_2d
        img_with_interference = np.clip(img_with_interference, 0, 1)

        return img_with_interference

    @staticmethod
    def plot_comparison(original, with_interference, signal_info=""):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        im1 = ax1.imshow(original, cmap='gray', vmin=0, vmax=1)
        ax1.set_title('Оригинальное РГГ')
        ax1.axis('off')

        ax2 = axes[1]
        im2 = ax2.imshow(with_interference, cmap='gray', vmin=0, vmax=1)
        ax2.set_title('РГГ с помеховым сигналом')
        ax2.axis('off')

        if signal_info:
            plt.suptitle(signal_info, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()