import os
import time
import json
from argparse import ArgumentParser
from subprocess import check_output
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

import plenoptic as po

from ppm.utils import init_data, init_model, PSNR, SSIM
from ppm.display import (
    visualize_prediction_dynamics, save_filter_viz, save_filter_anim
)


def parse_args():
    parser = ArgumentParser()

    # Model parameters
    parser.add_argument(
        "--model", type=str,
        help="""baselines: 'C', 'cMC', 'Spyr',
                polar prediction: 'PP', 'mPP',
                quadratic prediction:  'QP', 'mQP',
                Neural Net: 'CNN', 'Unet',
                more baselines: 'F', 'LE', 'L', 'LNL',
             """
    )
    parser.add_argument(
        "--filter-size", type=int, default=17,
        help="Filter size"
    )
    parser.add_argument(
        "--num-channels", type=int, default=32,
        help="Number of channels"
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Downsampling factor"
    )
    parser.add_argument(
        "--crop", type=int, default=-1,
        help="Crop spatially, default f // d"
    )
    parser.add_argument(
        "--pad-mode", type=str, default="valid",
        help="spatial convolution mode: 'valid' or 'same' (zero padding)",
    )
    parser.add_argument(
        "--branch", type=str, default="phase",
        help="""Choice of architecture (i.e. prediction mechanism):
                for PP: phase / phase_amp / phase_logamp / conj / proj
                for Controlnet: residual, vanilla""",
    )
    parser.add_argument(
        "--group-size", type=int, default=2,
        help="For PP: group size, default is 2 for quadrature pair",
    )
    parser.add_argument(
        "--num-quadratics", type=int, default=6,
        help="Number of quadratic units in learned predictor",
    )
    parser.add_argument(
        "--num-scales", type=int, default=5,
        help="""For Spyr: number of scales - 0 for auto;
                For Multiscale: number of scales (ie. number of downsamplings);
                For Unet TODO""",
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="""For controlnet: number of layers;
                For Unet: TODO
                For net: number of layers in Encoder (and Decoder) """,
    )
    parser.add_argument(
        "--epsilon", type=str, default="0",
        help="Epsilon of room, eg. 1e-10 (string to float)",
    )
    parser.add_argument(
        "--nonlin", type=str, default="amplitude",
        help="Activation function: amplitude, relu",
    )
    parser.add_argument(
        "--tied", type=int, default=1,
        help="""Tying the analysis and synthesis operator in PP""",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset", type=str, default="DAVIS",
        help="'PLANTED', 'VANH', 'DAVIS', 'UCF101' ",
    )
    parser.add_argument(
        "--image-size", type=int, default=128,
        help="Number of pixels. Default 128."
    )
    parser.add_argument(
        "--num-downs", type=int, default=1,
        help="Number of downsamplings, in [1, 2, 3]. Default 1.",
    )
    parser.add_argument(
        "--num-crops", type=int, default=1,
        help="Number of crops, in [1, 5, 9, 15, 21].",
    )
    parser.add_argument(
        "--fold", type=int, default=2017,
        help="""choice of train/test split:
                - 2017: predefined 60train-30test in ImageSets/2017/
                - [0-8]: nonoverlapping 80train-10test splits """,
    )
    parser.add_argument(
        "--normalize", type=int, default=1,
        help="""If True: mean subtract and div by std each channel,
                         using precomputed values
                Else: datarange in [0-1] """,
    )
    parser.add_argument(
        "--gray", type=int, default=1,
        help="If True, remove color channel, else keep RGB",
    )
    parser.add_argument(
        "--subset", "-u", type=int, default=0,
        help="Only use a subset of train clips (number of train samples)"
    )
    parser.add_argument(
        "--transform", type=str, default="translate",
        help="'translate', 'rotate', 'translate_rotate', 'translate_open' ",
    )

    # Optimization parameters
    parser.add_argument(
        "--overfit", type=int, default=0,
        help="If True, train on a single minibatch."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Choice of seed for random number generators.",
    )
    parser.add_argument(
        "--train", type=int, default=1,
        help="train or simply evaluate."
    )
    parser.add_argument(
        "--num-epochs", "-n", type=int, default=200,
        help="Number of epochs"
    )
    parser.add_argument(
        "--mini-batch", type=int, default=4,
        help="Number of elements in mini-batch",
    )
    parser.add_argument(
        "--learning-rate", "-l", type=float, default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--scheduler", type=str, default="plateau",
        help="step, plateau, cosine"
    )
    parser.add_argument(
        "--weight-decay", "-w", type=float, default=0,
        help="Weight decay"
    )
    parser.add_argument(
        "--max-grad-norm", "-x", type=float, default=0.5,
        help="Gradient clipping"
    )

    # Logging parameters
    parser.add_argument(
        "--output-dir", type=str,
        help="Directory for training logs",
    )
    parser.add_argument(
        "--log-every", type=int, default=10,
        help="Number of training epochs btw logs",
    )
    parser.add_argument(
        "--viz-pred-dynamics", type=int, default=0,
        help="log a test image and prediction in tensorboard during training",
    )

    args = parser.parse_args()

    # handle some defaults
    args.epsilon = float(args.epsilon)
    args.learning_rate = float(args.learning_rate)
    args.weight_decay = float(args.weight_decay)
    if args.crop == -1:
        args.crop = args.filter_size // args.stride

    return args


def init_logging(args):
    home = os.path.expanduser("~")
    output_dir = os.path.join(
        home + "/PolarPrediction/checkpoints/", args.output_dir
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    return output_dir


def init_optimizer(args, model):
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    if args.scheduler == "step":
        n_scheduler_steps = 5
        milestones = np.linspace(
            args.num_epochs // 2,
            args.num_epochs,
            n_scheduler_steps,
            dtype=int,
            endpoint=False,
        )
        print("milestones: ", milestones)  # 100 [50, 60, 70, 80, 90]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(milestones), gamma=0.5
        )
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=1 / 2, patience=2, min_lr=1e-5,
            verbose=True
        )
    elif args.scheduler == "cosine":
        # T_0 = args.num_epochs == just one cycle down
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-5, last_epoch=-1,
            verbose=True
        )
    return optimizer, scheduler


def train(model, train_dataloader, optimizer, device, data_stats,
          max_grad_norm=0.5):
    model.train()
    m = []
    p = []
    s = []
    for i, x in enumerate(tqdm(train_dataloader)):
        x = x.to(device)
        x, x_hat = model.predict(x)
        loss = F.mse_loss(x, x_hat)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"encountered loss {loss} at iter {i}")
            break
        m.append(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm > 0:
            # clipping gradients to mitigate spikes in the loss curve
            clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        with torch.no_grad():
            # [B T] -> [(BxT)]
            psnr = PSNR(x, x_hat, data_stats=data_stats).flatten()
            ssim = SSIM(x, x_hat, data_stats=data_stats).flatten()
            p.append(psnr)
            s.append(ssim)
    train_metrics = {}
    train_metrics["mse"] = torch.stack(m).mean().item()
    train_metrics["psnr"] = torch.stack(p).mean().item()
    train_metrics["ssim"] = torch.stack(s).mean().item()
    return train_metrics


@torch.no_grad()
def evaluate(model, test_dataloader, device, data_stats, reduce=True):
    """
    reduce: if True return average test loss / psnr
            else return test loss / psnr per image
                of shape [N, T] averaged over channel and space
    """
    model.eval()
    m = []
    p = []
    s = []
    for i, x in enumerate(tqdm(test_dataloader)):
        x = x.to(device)
        x, x_hat = model.predict(x)
        mse = (x_hat - x).pow(2).mean(dim=(1, 3, 4))  # [B T]
        loss = mse.mean()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"encountered loss {loss} at iter {i}")
            break
        psnr = PSNR(x, x_hat, data_stats=data_stats)  # [B T]
        ssim = SSIM(x, x_hat, data_stats=data_stats)  # [B T]
        m.append(mse)
        p.append(psnr)
        s.append(ssim)
    # [N B T] -> [(NxB) T]
    loss_test = torch.cat(m, dim=0).detach().cpu()
    psnr_test = torch.cat(p, dim=0).detach().cpu()
    ssim_test = torch.cat(s, dim=0).detach().cpu()
    if reduce:
        loss_test = loss_test.mean().item()
        psnr_test = psnr_test.mean().item()
        ssim_test = ssim_test.mean().item()
    test_metrics = {}
    test_metrics["mse"] = loss_test
    test_metrics["psnr"] = psnr_test
    test_metrics["ssim"] = ssim_test
    return test_metrics


def log(args, model, x_train, x_test, device, writer, epoch, test_metric,
        train_metric):
    if args.model in ["L", "LNL"]:
        w_in = model.W.weight.data
        f = w_in.shape[-1]
        t = w_in.shape[-3]
        fig = po.imshow(
            w_in.view(-1, 1, f, f),
            zoom=2,
            title=None,
            vrange="auto1",
            col_wrap=int(4 * t),
        )
        writer.add_figure("encoding weights", fig, epoch)

    elif args.model in ["PP", "mPP", "QP", "mQP"]:
        cw = np.minimum(8, args.num_channels * args.group_size)
        fig = save_filter_viz(
            model.W, args.group_size, freq=False, cw=cw, zoom=2, vrange="auto1"
        )
        writer.add_figure("encoding weights", fig, epoch)

    writer.add_scalars(
        "loss/",
        {"train": train_metric["mse"], "test": test_metric["mse"]},
        epoch
    )
    writer.add_scalars(
        "psnr/",
        {"train": train_metric["psnr"], "test": test_metric["psnr"]},
        epoch
    )
    writer.add_scalars(
        "ssim/",
        {"train": train_metric["ssim"], "test": test_metric["ssim"]},
        epoch
    )
    if args.viz_pred_dynamics:
        writer.add_figure(
            "prediction_train",
            visualize_prediction_dynamics(model, x_train, device),
            epoch,
        )
        writer.add_figure(
            "prediction_test",
            visualize_prediction_dynamics(model, x_test, device),
            epoch,
        )


def write_info(output_dir, train_time, test_time, num_parameters, epoch):
    """
    overwrites info to txt file

    TODO: frames/second
    """
    with open(os.path.join(output_dir, "info.txt"), "w") as text_file:
        train_time = int(np.array(train_time).mean().round())
        print(
            f"Average train time: {train_time//60}m {train_time%60}s",
            file=text_file,
        )
        test_time = int(np.array(test_time).mean().round())
        print(
            f"Average test time: {test_time//60}m {test_time%60}s",
            file=text_file,
        )
        print(f"Number of parameters: {num_parameters}", file=text_file)
        print(f"Epoch: {epoch}", file=text_file)


def main(args=None):
    if args is None:
        args = parse_args()
    print("\n \n", vars(args))

    output_dir = init_logging(args)
    writer = SummaryWriter(output_dir)

    # check device
    print(check_output(["hostname"]))
    print("cuda is available ", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    if args.model == "MC":
        device = "cpu"
    print("device: ", device)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"num threads {torch.get_num_threads()} \n")

    train_dataloader, test_dataloader, data_stats = init_data(args)
    assert len(train_dataloader) > 0

    # load a couple examples
    for x_train in train_dataloader:
        input_size = x_train.shape
        break
    for x_test in test_dataloader:
        break

    model = init_model(args)
    num_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("# params: ", num_parameters)
    if num_parameters > 0:
        print([p.shape for p in model.parameters() if p.requires_grad])

    model.to(device)

    print(
        "\ninput size ",
        input_size,
        "\nnum train mibi-batch ",
        len(train_dataloader),
        "\nnum test mibi-batch ",
        len(test_dataloader),
    )

    if args.train:
        train_time = []
        test_time = []

        optimizer, scheduler = init_optimizer(args, model)

        loss_train_best = np.inf
        loss_test_best = np.inf
        for epoch in range(args.num_epochs):

            print(epoch + 1, "/", args.num_epochs)
            tic = time.time()
            train_metric = train(
                model, train_dataloader, optimizer,
                device, data_stats, args.max_grad_norm,
            )
            toc = time.time()
            train_time.append(toc - tic)
            print("train", train_metric)

            if not hasattr(scheduler, "patience"):
                scheduler.step()
            # else: reducing lr on plateau
            #   ie. first evaluate test loss before stepping scheduler

            if (epoch + 1) % args.log_every == 0 or epoch == 0:

                if train_metric["mse"] < loss_train_best:
                    loss_train_best = train_metric["mse"]
                    print(f"new best train loss {loss_train_best} \n")
                    # save (and overwrite) model
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_dir, "model_train.pt")
                    )

                tic = time.time()
                test_metric = evaluate(
                    model, test_dataloader, device, data_stats
                )
                toc = time.time()
                test_time.append(toc - tic)
                print("test", test_metric)

                if args.scheduler == "plateau":
                    scheduler.step(metrics=test_metric["mse"])
                    # print("current lr: ",
                    # scheduler.optimizer.param_groups[0]['lr'])

                if test_metric["mse"] < loss_test_best:
                    loss_test_best = test_metric["mse"]
                    print(f"new best test loss {loss_test_best} \n")
                    # save (and overwrite) model
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_dir, "model.pt")
                    )

                else:
                    print(f"current best test loss {loss_test_best}")

                # tensorboard
                log(
                    args, model, x_train, x_test, device, writer, epoch + 1,
                    test_metric, train_metric,
                )
                write_info(
                    output_dir, train_time, test_time, num_parameters, epoch + 1
                )
        writer.flush()
        writer.close()

        if not args.overfit:
            if args.model in ["PP", "mPP"] and args.group_size == 2:
                cw = np.minimum(4, args.num_channels)
                save_filter_anim(
                    model.W, cw=cw, zoom=4,
                    path=os.path.join(output_dir, "filters")
                )

    else:  # simply evaluate, no training
        tic = time.time()
        train_metric = evaluate(model, train_dataloader, device, data_stats)
        toc = time.time()
        train_time = toc - tic
        print("train", train_metric)
        tic = time.time()
        test_metric = evaluate(model, test_dataloader, device, data_stats)
        toc = time.time()
        test_time = toc - tic
        print("test", test_metric)
        log(
            args, model, x_train, x_test, device, writer, 0,
            test_metric, train_metric
        )
        write_info(output_dir, train_time, test_time, num_parameters, 0)

if __name__ == "__main__":
    main()
