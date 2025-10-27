import argparse
from pathlib import Path

import torch

from models.LPR_model import LPR_model


def adapt_weights(old_state, new_model):
    """Adapt legacy decoder weights to match current layer shapes."""
    new_state = new_model.state_dict()

    for name, param in new_state.items():
        if "deconv" in name and "weight" in name:
            old_param = old_state[name]
            if old_param.shape != param.shape:
                # Simple center pad/crop to align kernel shapes
                if param.shape[-1] > old_param.shape[-1]:
                    new_param = torch.zeros_like(param)
                    pad = (param.shape[-1] - old_param.shape[-1]) // 2
                    new_param[
                        :, :, pad : pad + old_param.shape[-2], pad : pad + old_param.shape[-1]
                    ] = old_param
                    new_state[name] = new_param
                else:
                    pad = (old_param.shape[-1] - param.shape[-1]) // 2
                    new_state[name] = old_param[
                        :, :, pad : pad + param.shape[-2], pad : pad + param.shape[-1]
                    ]
            else:
                new_state[name] = old_param
        else:
            new_state[name] = old_state[name]

    new_model.load_state_dict(new_state)
    return new_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export the LPR model to ONNX format.")
    parser.add_argument(
        "-w",
        "--weights-path",
        type=str,
        required=True,
        help="Path to the trained model weights (.pth).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="lpr.onnx",
        help="Destination path for the exported ONNX model.",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=96,
        help="Input image width expected by the model.",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=32,
        help="Input image height expected by the model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version to target.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    alphabet = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航应急0123456789ABCDEFGHJKLMNPQRSTUVWXYZO"
    alphabet = "-" + alphabet

    model = LPR_model(
        nc=1,
        nclass=len(alphabet),
        imgW=args.img_width,
        imgH=args.img_height,
    ).to(device)

    state_dict = torch.load(args.weights_path, map_location=device, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    adapt_weights(state_dict, model)
    model.decoder.enable_onnx_export(True)
    model.eval()

    dummy_input = torch.zeros(1, 1, args.img_height, args.img_width, device=device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path.as_posix(),
            export_params=True,
            opset_version=args.opset,
            input_names=["images"],
            output_names=["probabilities"],
            # dynamic_axes={"images": {0: "batch"}, "probabilities": {0: "batch"}},
        )

    print(f"ONNX model exported to {output_path.resolve()}")


if __name__ == "__main__":
    main()
