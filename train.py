# This is a sample Python script.
import os.path
import argparse
import torch
import time
from model import build_model
from data import make_dataset
from loss import PerceptualLoss, L1Loss, ArtifactLoss
from utils import save_img, make_checkpoint_dir, setup_logger, log_csv, AverageMeter, ListAction
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs


def parser_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", '-i', type=str, default='./', help="path of inputs")
    parser.add_argument("--test_path", type=str, default=None, help="path of test set")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epoch")
    # optimizer
    parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
    # dataloader param
    parser.add_argument("--max_value", type=int, default=1, help="max value of reflectance")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--workers", type=int, default=16, help="workers of dataloader")
    # architecture param
    parser.add_argument("--in_channels", "-ic", type=int, default=3, help="number of input channels")
    parser.add_argument("--out_channels", "-oc", type=int, default=3, help="number of output channels")
    parser.add_argument("--num_features", type=int, default=32, help="feature numbers")
    parser.add_argument('--kernel_size', type=int, default=3, help='size of spatiotemporal kernel')
    parser.add_argument('--num_rrdb', type=int, default=8, help='number of RRDB')
    parser.add_argument('--num_msv', type=int, default=3, help='depth of multi scale volume')
    # loss param
    parser.add_argument('--channel_weight', type=str, action=ListAction, default=None,
                        help='the weight coefficient of bands')
    # save and load weight
    parser.add_argument("--state", type=str, default="", help="path of loaded weight")
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoint", help='output path of checkpoint')
    parser.add_argument("--interval", type=int, default=10, help="interval saving")

    opt = parser.parse_args()
    opt.is_train = True
    return opt


def train():
    opt = parser_option()
    # ===accelerate initialization===
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=1, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    opt.device = device
    # ========logger setup============
    make_checkpoint_dir(opt.checkpoint_dir)
    time_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(opt.checkpoint_dir, f'{time_name}.log')
    if accelerator.is_local_main_process: setup_logger(log_file=log_file)
    logger = get_logger(name='AGSDF')
    csv_file = os.path.join(opt.checkpoint_dir, f'{time_name}.csv')
    # ===========parameters==========
    params_dict = vars(opt)
    params_txt = "Params\n"
    for key, value in params_dict.items():
        key, value = str(key).ljust(15)[:15], str(value).ljust(50)[:50]
        params_txt = params_txt + f'{key}: {value} \n'
    logger.info(params_txt)
    # ===========dataset=============
    train_dataloader = make_dataset(opt)
    logger.info('training dataset setup ...')
    val_dataloader = make_dataset(opt, is_val=True) if opt.test_path is not None else None
    logger.info('validation dataset setup ...')
    # ============model==============
    model_src, is_load = build_model(opt)
    if is_load: logger.info('model load success ...')
    model_src = model_src.to(device)
    net_params = sum(map(lambda x: x.numel(), model_src.parameters()))
    logger.info(f"Network with parameters: {net_params}")
    optimizer = torch.optim.Adam(model_src.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    # ============losses=============
    l_loss = L1Loss(opt.channel_weight).to(device)
    a_loss = ArtifactLoss().to(device)
    p_loss = PerceptualLoss(opt.out_channels).to(device)
    logger.info(f'All loss setup ...')
    # accelerator
    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model_src, optimizer
    )
    for epoch in range(1, opt.epochs + 1):
        epoch_pix_loss = AverageMeter()
        epoch_perc_loss = AverageMeter()
        epoch_arti_loss = AverageMeter()
        epoch_total_loss = AverageMeter()
        for i, data in enumerate(train_dataloader):
            c1 = data["c1"].float().to(device)
            c2 = data["c2"].float().to(device)
            f1 = data["f1"].float().to(device)
            f2 = data["f2"].float().to(device)
            optimizer.zero_grad()
            output = model(c1, f1, c2)
            # batch loss calculation
            pix_loss = l_loss(f2, output)
            perc_loss = p_loss(f2, output)
            arti_loss = a_loss(f2, output, c1, c2, f1)
            loss_total = pix_loss + perc_loss * 0.5 + arti_loss * 10
            # update epoch loss
            epoch_pix_loss.update(pix_loss.item())
            epoch_perc_loss.update(perc_loss.item())
            epoch_arti_loss.update(arti_loss.item())
            epoch_total_loss.update(loss_total.item())
            # backward
            accelerator.backward(loss_total)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
        # logger
        logger.info("[Epoch %d/%d]-[PixelLoss:%f]-[PerceptualLoss:%f]-[ArtifactLoss:%f]"
                    % (epoch, opt.epochs, epoch_pix_loss.avg, epoch_perc_loss.avg, epoch_arti_loss.avg))
        csv_value = [epoch, epoch_pix_loss.avg, epoch_perc_loss.avg, epoch_arti_loss.avg, epoch_total_loss.avg]
        csv_header = ['epoch', 'pixel_loss', 'perceptual_loss', 'artifact_loss', 'total_loss']
        if accelerator.is_local_main_process: log_csv(csv_file, csv_value, header=csv_header)

        # model and val image saving
        if opt.interval != -1 and epoch % opt.interval == 0:
            # Save model checkpoints
            accelerator.wait_for_everyone()
            save_model_path = os.path.join(opt.checkpoint_dir, "models", f"{epoch}.pth")
            accelerator.save(accelerator.unwrap_model(model).state_dict(), save_model_path)
            # save val images
            if val_dataloader is not None and accelerator.is_local_main_process:
                save_img_path = os.path.join(opt.checkpoint_dir, 'visual')
                with torch.no_grad():
                    for val_data in val_dataloader:
                        c1 = val_data['c1'].float().to(device)
                        c2 = val_data["c2"].float().to(device)
                        f1 = val_data["f1"].float().to(device)
                        val_out = model_src(c1, f1, c2)
                        save_img(val_data, val_out, save_img_path, epoch, 'val')

    # save late model
    accelerator.wait_for_everyone()
    save_path = os.path.join(opt.checkpoint_dir, "models", "latest.pth")
    accelerator.save(accelerator.unwrap_model(model).state_dict(), save_path)


if __name__ == "__main__":
    # accelerate launch train.py --args
    """
    
    """
    train()
