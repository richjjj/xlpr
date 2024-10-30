from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import lmdb
import torch
from PIL import Image

import random

from straug.noise import GaussianNoise, SpeckleNoise
from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
from straug.process import Posterize, Equalize, AutoContrast, Sharpness, Color

import sys


class LPR_LMDB_Dataset(Dataset):
    def __init__(
        self,
        root,
        K=8,
        use_ratio=1.0,
        isdegrade=False,
        isAug=False,
    ):
        self.root = root
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            print("cannot create lmdb from %s" % (root))
            sys.exit(0)

        self.isdegrade = isdegrade
        self.K = K
        self.isAug = isAug

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get("num-samples".encode()))
            self.nSamples = int(nSamples * use_ratio)
            self.filtered_index_list = [index + 1 for index in range(self.nSamples)]

    def __len__(self):
        return self.nSamples

    def Image2Numpy(self, pil_image):
        np_image = np.array(pil_image).astype("float32")
        np_image = np_image / 255
        np_image = np_image.reshape(1, np_image.shape[0], np_image.shape[1])

        return np_image

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = "label-%09d".encode() % index
            label = txn.get(label_key).decode("utf-8")  # string
            img_key = "image-%09d".encode() % index
            image = txn.get(img_key)  # buf raw image
            image_array = np.frombuffer(image, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
            # cv2.imwrite("a.jpg", img)
            image = Image.fromarray(img)
            # 将图像转换为灰度
            image_gray = image.convert("L")
            image_gray = image_gray.resize((96, 32))
            if self.isAug:
                aug_img = augment(image_gray, True, 96, 32)
                np_aug_img = self.Image2Numpy(aug_img)

                return (np_aug_img, label)

            image_array = self.Image2Numpy(image_gray)
            if self.isdegrade:
                if random.random() < 0.05:
                    image_array = cv2.resize(image_gray, (64, 24))
                    # print('image_array',image_array.shape)

                    image_array = cv2.resize(image_array, (96, 32))
                    image_array = (np.reshape(image_array, (32, 96, 1))).transpose(
                        2, 0, 1
                    )

            return (image_array, label)


def Perspective(img, mag=-1, prob=1.0):
    rng = np.random.default_rng()

    if rng.uniform(0, 1) > prob:
        return img

    w, h = img.size

    # upper-left, upper-right, lower-left, lower-right
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # low = 0.3

    # b = [.05, .1, .15]
    b = [0.1, 0.15, 0.2]
    if mag < 0 or mag >= len(b):
        index = 2
    else:
        index = mag
    low = b[index]

    high = 1 - low

    if rng.uniform(0, 1) > 0.5:
        topright_y = rng.uniform(low, low + 0.1) * h
        topright_x = rng.uniform(high + low / 2, 1) * w
        bottomleft_x = rng.uniform(0, low / 2) * w
        bottomleft_y = rng.uniform(high - 0.1, high) * h
        dest = np.float32(
            [[0, 0], [topright_x, topright_y], [bottomleft_x, bottomleft_y], [w, h]]
        )
    else:
        topleft_y = rng.uniform(low, low + 0.1) * h
        topleft_x = rng.uniform(0, low / 2) * w
        bottomright_y = rng.uniform(high - 0.1, high) * h
        bottomright_x = rng.uniform(high + low / 2, 1) * w
        dest = np.float32(
            [[topleft_x, topleft_y], [w, 0], [0, h], [bottomright_x, bottomright_y]]
        )

    M = cv2.getPerspectiveTransform(src, dest)
    img = np.asarray(img)
    img = cv2.warpPerspective(img, M, (w, h))
    img = Image.fromarray(img)

    return img


def augment(PIL_img, isPIL=True, imgW=96, imgH=32):
    noise_switch = {"0": GaussianNoise(), "1": SpeckleNoise()}

    camera_switch = {
        "0": Contrast(),
        "1": Brightness(),
        "2": JpegCompression(),
        "3": Pixelate(),
    }
    process_switch = {
        "0": Posterize(),
        "1": Equalize(),
        "2": AutoContrast(),
        "3": Sharpness(),
        "4": Color(),
    }

    if isPIL:
        img = PIL_img
    else:
        img = Image.fromarray(PIL_img)  # 转为PIL Image格式

    switch = random.randint(0, 2)
    # switch = 5

    if switch == 0:
        img = noise_switch[str(random.randint(0, 1))](img, mag=0)
    elif switch == 1:
        img = camera_switch[str(random.randint(0, 3))](img, mag=0)
    elif switch == 2:
        img = process_switch[str(random.randint(0, 4))](img, mag=0)

    # img = Perspective(img, mag=0,  prob=0.5)
    img = Perspective(img, mag=2, prob=0.5)
    # img = Rotate()(img, mag=0,  prob=0.4)

    if not isPIL:
        img = np.asarray(img)

    return img


if __name__ == "__main__":
    dataset = LPR_LMDB_Dataset("../clpr_lmdb/val", use_ratio=0.1, isAug=True)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for i_batch, (image, label) in enumerate(dataloader):
        aug_img = torch.reshape(image[0], (32, 96))
        aug_img = aug_img.numpy() * 255
        cv2.imwrite("rb3_aug3.png", aug_img)
