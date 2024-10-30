from __future__ import print_function
from torch.utils.data import DataLoader
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import os
import utils
import models.LPR_model as model
import importlib.util
import shutil

# from dataset_own import LPRDataset
from dataset_lmdb import LPR_LMDB_Dataset


# custom weights initialization called on lpr_model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val(lpr_model, val_loader, criterion, iteration, max_i=1000, message="Val_acc"):
    print("Start val")
    for p in lpr_model.parameters():
        p.requires_grad = False
    lpr_model.eval()
    i = 0
    n_correct = 0

    for i_batch, (image, label) in enumerate(val_loader):
        image = image.to(device)
        preds = lpr_model(image)

        batch_size = image.size(0)
        cost = 0
        preds_all = preds
        preds = torch.chunk(preds, preds.size(1), 1)

        text = converter.encode_list(label, K=params.K)
        text = text.to(device)
        for i, item in enumerate(preds):
            item = item.squeeze()
            gt = text[:, i]
            cost += criterion(item, gt) / batch_size

        _, preds_all = preds_all.max(2)
        sim_preds = converter.decode_list(preds_all.data)
        text_label = label
        for pred, target in zip(sim_preds, text_label):
            pred = pred.replace("-", "")
            if pred == target:
                n_correct += 1

        if (i_batch + 1) % params.displayInterval == 0:
            print(
                "[%d/%d][%d/%d] loss: %f"
                % (iteration, params.niter, i_batch, len(val_loader), cost.data)
            )

        if i_batch == max_i:
            break
    for raw_pred, pred, gt in zip(preds_all, sim_preds, text_label):
        raw_pred = raw_pred.data
        pred = pred.replace("-", "")
        print("raw_pred: %-20s, pred: %-20s, gt: %-20s" % (raw_pred, pred, gt))

    print(n_correct)
    print(i_batch * params.val_batchSize)
    if max_i * params.val_batchSize < len(val_dataset):
        accuracy = n_correct / (max_i * params.val_batchSize)
    else:
        accuracy = n_correct / len(val_dataset)
    print(message + " accuray: %f" % (accuracy))

    return accuracy


def train(lpr_model, train_loader, criterion, iteration):
    for p in lpr_model.parameters():
        p.requires_grad = True
    lpr_model.train()

    for i_batch, (image, label) in enumerate(train_loader):
        image = image.to(device)  # image：tensor[b, 1, 32, 96]
        text = converter.encode_list(label, K=params.K)  # tensor[b, 8]
        text = text.to(device)
        preds = lpr_model(image)

        cost = 0
        preds = torch.chunk(preds, preds.size(1), 1)
        for i, item in enumerate(preds):
            item = item.squeeze()
            gt = text[:, i]
            cost_item = criterion(item, gt)
            cost += cost_item

        lpr_model.zero_grad()
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)
        if (i_batch + 1) % params.displayInterval == 0:
            print(
                "[%d/%d][%d/%d] Loss: %f lr: %f"
                % (
                    iteration,
                    params.niter,
                    i_batch,
                    len(train_loader),
                    loss_avg.val(),
                    scheduler.get_last_lr()[0],
                )
            )
            loss_avg.reset()


def main(lpr_model, train_loader, val_loader, criterion):
    lpr_model = lpr_model.to(device)
    criterion = criterion.to(device)
    Iteration = 0
    best_acc = 0
    best_epoch = 0
    while Iteration < params.niter:
        train(lpr_model, train_loader, criterion, Iteration)

        accuracy = val(
            lpr_model, val_loader, criterion, Iteration, max_i=params.valInterval
        )

        for p in lpr_model.parameters():
            p.requires_grad = True

        if Iteration % params.saveInterval == 0:
            torch.save(
                lpr_model.state_dict(),
                f"{params.experiment}/latest.pth",
            )
        print(
            f"current accuracy: {accuracy}/{Iteration}; best accuracy : {best_acc}/{best_epoch}"
        )
        if accuracy > best_acc:
            best_path = f"{params.experiment}/best_model.pth"
            torch.save(
                lpr_model.state_dict(),
                best_path,
            )
            best_acc = accuracy
            best_epoch = Iteration

        scheduler.step()
        Iteration += 1
    shutil.copy(
        f"{params.experiment}/best_model.pth",
        f"{params.experiment}/best_model_{best_epoch}_{best_acc}.pth",
    )


def backward_hook(module, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0  # replace all nan/inf in gradients to zero


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPR Model Training Parameters")
    parser.add_argument("-p", "--params_path", type=str, default="configs/plate.py")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("params", args.params_path)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    if not os.path.exists(params.experiment):
        os.makedirs(params.experiment)
    # manualSeed = 1111
    # random.seed(manualSeed)
    # np.random.seed(manualSeed)
    # torch.manual_seed(manualSeed)
    # torch.cuda.manual_seed(manualSeed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    device = torch.device(
        "cuda:" + str(params.gpu) if torch.cuda.is_available() else "cpu"
    )

    dataset = LPR_LMDB_Dataset(params.train_data, isAug=True, isdegrade=True)
    val_dataset = LPR_LMDB_Dataset(params.val_data, isAug=False, isdegrade=False)

    train_loader = DataLoader(
        dataset, batch_size=params.batchSize, shuffle=True, num_workers=params.workers
    )
    print("train_loader", len(train_loader))
    # shuffle=True, just for time consuming.
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.val_batchSize,
        shuffle=False,
        num_workers=params.workers,
    )

    converter = utils.strLabelConverter(params.alphabet)
    nclass = len(params.alphabet) + 1
    nc = params.nc

    criterion = torch.nn.CrossEntropyLoss()

    # cnn or others
    if params.model_type == "CNN":
        lpr_model = model.LPR_model(
            nc, nclass, imgW=params.imgW, imgH=params.imgH, K=params.K
        ).to(device)
    else:
        print("未实现")
        pass
    if params.pre_model != "":
        print("loading pretrained model from %s" % params.pre_model)
        lpr_model.load_state_dict(torch.load(params.pre_model, map_location=device))
    else:
        lpr_model.apply(weights_init)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if params.adam:
        optimizer = optim.Adam(
            lpr_model.parameters(), lr=params.lr, betas=(params.beta1, 0.999)
        )
    elif params.adadelta:
        optimizer = optim.Adadelta(lpr_model.parameters(), lr=params.lr)
    else:
        optimizer = optim.RMSprop(lpr_model.parameters(), lr=params.lr)

    # TODO 修改学习率策略
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=params.lr_step, gamma=0.8
    )
    # lpr_model.register_backward_hook(backward_hook)
    main(lpr_model, train_loader, val_loader, criterion)
