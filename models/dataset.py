from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import lmdb
import torch
from PIL import Image
import random
from typing import Tuple, List, Union, Dict
from dataclasses import dataclass
from enum import Enum


class AugmentationType(Enum):
    NOISE = "noise"
    CAMERA = "camera"
    PROCESS = "process"


class ImageAugmentor:
    """图像增强处理类"""

    def __init__(self):
        from straug.noise import GaussianNoise, SpeckleNoise
        from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
        from straug.process import Posterize, Equalize, AutoContrast, Sharpness, Color

        self.augmentation_methods = {
            AugmentationType.NOISE: {0: GaussianNoise(), 1: SpeckleNoise()},
            AugmentationType.CAMERA: {
                0: Contrast(),
                1: Brightness(),
                2: JpegCompression(),
                3: Pixelate(),
            },
            AugmentationType.PROCESS: {
                0: Posterize(),
                1: Equalize(),
                2: AutoContrast(),
                3: Sharpness(),
                4: Color(),
            },
        }

    def apply_perspective(
        self, img: Image.Image, mag: int = 2, prob: float = 0.5
    ) -> Image.Image:
        """应用透视变换"""
        rng = np.random.default_rng()
        if rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        b = [0.1, 0.15, 0.2]
        index = 2 if mag < 0 or mag >= len(b) else mag
        low = b[index]
        high = 1 - low

        if rng.uniform(0, 1) > 0.5:
            dest = self._get_right_perspective(w, h, low, high, rng)
        else:
            dest = self._get_left_perspective(w, h, low, high, rng)

        M = cv2.getPerspectiveTransform(src, dest)
        img_array = np.asarray(img)
        img_array = cv2.warpPerspective(img_array, M, (w, h))
        return Image.fromarray(img_array)

    def _get_right_perspective(
        self, w: int, h: int, low: float, high: float, rng
    ) -> np.ndarray:
        topright_y = rng.uniform(low, low + 0.1) * h
        topright_x = rng.uniform(high + low / 2, 1) * w
        bottomleft_x = rng.uniform(0, low / 2) * w
        bottomleft_y = rng.uniform(high - 0.1, high) * h
        return np.float32(
            [[0, 0], [topright_x, topright_y], [bottomleft_x, bottomleft_y], [w, h]]
        )

    def _get_left_perspective(
        self, w: int, h: int, low: float, high: float, rng
    ) -> np.ndarray:
        topleft_y = rng.uniform(low, low + 0.1) * h
        topleft_x = rng.uniform(0, low / 2) * w
        bottomright_y = rng.uniform(high - 0.1, high) * h
        bottomright_x = rng.uniform(high + low / 2, 1) * w
        return np.float32(
            [[topleft_x, topleft_y], [w, 0], [0, h], [bottomright_x, bottomright_y]]
        )

    def augment(self, img: Image.Image) -> Image.Image:
        aug_type = random.choice(list(AugmentationType))
        methods = self.augmentation_methods[aug_type]
        method_idx = random.randint(0, len(methods) - 1)

        img = methods[method_idx](img, mag=0)
        img = self.apply_perspective(img, mag=2, prob=0.5)
        return img


@dataclass
class DatasetConfig:
    root_path: Union[str, List[str]]
    use_ratio: float = 1.0
    enable_augmentation: bool = False
    enable_degrade: bool = False
    img_width: int = 96
    img_height: int = 32

    def __post_init__(self):
        if isinstance(self.root_path, str):
            self.root_path = [self.root_path]


class LPRDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.envs = self._init_lmdbs()
        self.augmentor = ImageAugmentor() if config.enable_augmentation else None
        self.samples_indices = self._init_samples()

    def _init_lmdbs(self) -> Dict[str, lmdb.Environment]:
        envs = {}
        for path in self.config.root_path:
            env = lmdb.open(
                path,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            if not env:
                raise RuntimeError(f"无法创建LMDB: {path}")
            envs[path] = env
        return envs

    def _init_samples(self) -> List[Tuple[str, int]]:
        all_samples = []

        for path, env in self.envs.items():
            with env.begin(write=False) as txn:
                total_samples = int(txn.get("num-samples".encode()))
                num_samples = int(total_samples * self.config.use_ratio)
                samples = [
                    (path, idx) for idx in range(1, num_samples + 1)
                ]  # 创建的lmdb里所有从1开始
                all_samples.extend(samples)

        return all_samples

    def __len__(self) -> int:
        return len(self.samples_indices)

    def _image_to_numpy(self, pil_image: Image.Image) -> np.ndarray:
        np_image = np.array(pil_image).astype("float32") / 255
        return np_image.reshape(1, np_image.shape[0], np_image.shape[1])

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        path, sample_idx = self.samples_indices[index]

        with self.envs[path].begin(write=False) as txn:
            # 获取标签
            label_key = f"label-{sample_idx:09d}".encode()
            label = txn.get(label_key).decode("utf-8")

            # 获取图像
            img_key = f"image-{sample_idx:09d}".encode()
            img_data = txn.get(img_key)
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            image = Image.fromarray(img)

            image_gray = image.convert("L")
            image_gray = image_gray.resize(
                (self.config.img_width, self.config.img_height)
            )

            if self.config.enable_augmentation:
                image_gray = self.augmentor.augment(image_gray)
            image_array = self._image_to_numpy(image_gray)
            # if self.config.enable_degrade and random.random() < 0.05:
            #     image_array = self._apply_degrade(image_array)

            return image_array, label

    def _apply_degrade(self, image_array: np.ndarray) -> np.ndarray:
        """应用图像降质"""
        degraded = cv2.resize(image_array, (64, 24))
        degraded = cv2.resize(degraded, (self.config.img_width, self.config.img_height))
        return (
            np.reshape(degraded, (self.config.img_height, self.config.img_width, 1))
        ).transpose(2, 0, 1)


def create_dataloader(
    root_path: str,
    batch_size: int = 32,
    use_ratio: float = 1.0,
    image_width: int = 96,
    image_height: int = 32,
    enable_augmentation: bool = False,
    enable_degrade: bool = False,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    config = DatasetConfig(
        root_path=root_path,
        use_ratio=use_ratio,
        enable_degrade=enable_degrade,
        enable_augmentation=enable_augmentation,
        img_width=image_width,
        img_height=image_height,
    )
    dataset = LPRDataset(config)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


if __name__ == "__main__":
    # 示例用法
    dataloader = create_dataloader(
        root_path="../clpr_lmdb/val",
        batch_size=2,
        use_ratio=0.1,
        enable_augmentation=True,
    )

    for i_batch, (image, label) in enumerate(dataloader):
        aug_img = torch.reshape(image[0], (32, 96))
        aug_img = aug_img.numpy() * 255
        cv2.imwrite(f"augmented_sample_{i_batch}.png", aug_img)
        break  # 只保存第一个批次的图像
