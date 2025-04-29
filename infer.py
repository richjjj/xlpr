import models.LPR_model as model
import argparse
import torch
import cv2
import numpy as np


def infer(lpr_model, image_path, device):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (96, 32))

    np_array = image.astype("float32") / 255.0
    np_array = np_array.reshape(1, 1, 32, 96)
    input = torch.from_numpy(np_array).to(device)
    output = lpr_model(input)
    _, preds = output.max(2)

    return preds.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--weights_path",
        type=str,
        default="output_1/best_model.pth",
    )
    parser.add_argument("-i", "--image_path", type=str, default="imgs/1.jpg")
    args = parser.parse_args()
    alphabet = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航应急0123456789ABCDEFGHJKLMNPQRSTUVWXYZO"
    alphabet = "-" + alphabet

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    lpr_model = model.LPR_model(nc=1, nclass=len(alphabet)).to(device)
    state_dict = torch.load(args.weights_path, weights_only=True, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    lpr_model.load_state_dict(
        state_dict
    )
    lpr_model.eval()

    image_path = args.image_path
    preds = infer(lpr_model, image_path, device)
    print(preds)
    text = "".join(alphabet[p] for p in preds[0] if p != 0)
    print(text)
