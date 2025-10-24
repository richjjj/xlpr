import models.LPR_model as model
import argparse
import torch
import cv2
import numpy as np
import math

def adapt_weights(old_state, new_model):
    new_state = new_model.state_dict()

    for name, param in new_state.items():
        if 'deconv' in name and 'weight' in name:
            old_param = old_state[name]
            if old_param.shape != param.shape:
                # 权重尺寸适配（简单的中心填充或裁剪）
                if param.shape[-1] > old_param.shape[-1]:  # 需要填充
                    # 中心填充
                    new_param = torch.zeros_like(param)
                    pad = (param.shape[-1] - old_param.shape[-1]) // 2
                    new_param[:, :, pad:pad+old_param.shape[-2], pad:pad+old_param.shape[-1]] = old_param
                    new_state[name] = new_param
                else:  # 需要裁剪
                    # 中心裁剪
                    pad = (old_param.shape[-1] - param.shape[-1]) // 2
                    new_state[name] = old_param[:, :, pad:pad+param.shape[-2], pad:pad+param.shape[-1]]
            else:
                new_state[name] = old_param
        else:
            new_state[name] = old_state[name]

    new_model.load_state_dict(new_state)
    return new_model

def infer(lpr_model, image_path, device):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (96, 32))

    np_array = image.astype("float32") / 255.0
    np_array = np_array.reshape(1, 1, 32, 96)
    input = torch.from_numpy(np_array).to(device)
    with torch.no_grad():
        output = lpr_model(input)
        probabilities = torch.softmax(output, dim=2)
        confidences, preds = probabilities.max(2)

    return preds.detach().cpu().numpy(), confidences.detach().cpu().numpy()


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
    # attention没修改的话 则不用调整权重。只是为了适配rknn框架
    adapt_weights(state_dict, lpr_model) 
    # lpr_model.load_state_dict(state_dict)
    lpr_model.eval()

    image_path = args.image_path
    preds, confidences = infer(lpr_model, image_path, device)
    print(preds)
    plate_chars = []
    char_confidences = []
    for idx, p in enumerate(preds[0]):
        if p != 0:
            plate_chars.append(alphabet[p])
            char_confidences.append(float(confidences[0][idx]))
    text = "".join(plate_chars)
    print(text)
    if char_confidences:
        plate_confidence = float(math.prod(char_confidences))
        print("Char confidences:", " ".join(f"{c:.4f}" for c in char_confidences))
        print(f"Plate confidence: {plate_confidence:.4f}")
    else:
        print("No valid characters detected.")
