import os
import time
import datetime

import torch

from src import deeplabv3_resnet50
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from camvid_dataset import get_camvid_dataset
import numpy as np


def create_model(aux, num_classes, pretrain=True):
    model = deeplabv3_resnet50(aux=aux, num_classes=num_classes)

    if pretrain:
        weights_dict = torch.load(
            "deeplabv3_resnet50_coco.pth", map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(
            weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def main(args):
    dataset_path = args.dataset_path
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # print("Using {} dataloader workers".format(num_workers))
    train_dataset, valid_dataset = get_camvid_dataset(dataset_path)
    # length of dataset
    print(f"train dataset length: {len(train_dataset)}")
    print(f"valid dataset length: {len(valid_dataset)}")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               #    collate_fn=train_dataset.collate_fn
                                               )

    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             #  collate_fn=valid_dataset.collate_fn
                                             )

    model = create_model(
        aux=args.aux, num_classes=num_classes, pretrain=args.pretrain)
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.parameters()
                  if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(
        optimizer, len(train_loader), args.epochs, warmup=True)

    # import matplotlib.pyplot as plt
    # lr_list = []
    # for _ in range(args.epochs):
    #     for _ in range(len(train_loader)):
    #         lr_scheduler.step()
    #         lr = optimizer.param_groups[0]["lr"]
    #         lr_list.append(lr)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    # metric list
    global_acc_list = []
    iou_list = []
    train_loss_list = []
    val_loss_list = []
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        val_loss, confmat = evaluate(model, val_loader, device=device,
                           num_classes=num_classes)
        acc, _, iou = confmat.compute()
        val_info = str(confmat)
        print(val_info)
        # append metric
        global_acc_list.append(acc.item() * 100)
        iou_list.append(iou.mean().item() * 100)
        train_loss_list.append(mean_loss)
        val_loss_list.append(val_loss)

        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        # save list to npy
        np.save(f"./results/{args.name}/global_acc_list.npy",
                np.array(global_acc_list))
        np.save(f"./results/{args.name}/iou_list.npy", np.array(iou_list))
        np.save(f"./results/{args.name}/train_loss_list.npy",
                np.array(train_loss_list))
        np.save(f"./results/{args.name}/val_loss_list.npy",
                np.array(val_loss_list))


    torch.save(save_file, f"save_weights/{args.name}.pth")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 training")
    parser.add_argument(
        "--dataset-path", default="./CamVid", help="dataset path")
    parser.add_argument("--num-classes", default=32, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10,
                        type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    # pretrain
    parser.add_argument("-p", "--pretrain",
                        action="store_true", help="use pretrain weights")
    # experiment name
    parser.add_argument("-n", "--name", default="deeplabv3",
                        type=str, help="experiment name")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    # result dir
    if not os.path.exists("./results"):
        os.mkdir("./results")
    # exp name dir under result
    if not os.path.exists(f"./results/{args.name}"):
        os.mkdir(f"./results/{args.name}")
    main(args)
