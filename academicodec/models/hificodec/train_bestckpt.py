import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from academicodec.models.hificodec.env import AttrDict, build_env
from academicodec.models.hificodec.meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from academicodec.models.encodec.msstftd import MultiScaleSTFTDiscriminator
from academicodec.models.hificodec.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from academicodec.models.hificodec.models import feature_loss, generator_loss, discriminator_loss
from academicodec.models.hificodec.models import Encoder, Quantizer
from academicodec.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

def save_best_checkpoint(checkpoint_path, model_state, identifier):
    """ Save best model separately. """
    best_path = os.path.join(checkpoint_path, f"best_{identifier}")
    save_checkpoint(best_path, model_state, num_ckpt_keep=1)
    print(f"✅ Best {identifier} model updated at {best_path}")

def train(rank, a, h):
    torch.cuda.set_device(rank)
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    encoder = Encoder(h).to(device)
    generator = Generator(h).to(device)
    quantizer = Quantizer(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    mstftd = MultiScaleSTFTDiscriminator(32).to(device)

    if rank == 0:
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("Checkpoints directory:", a.checkpoint_path)

    cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
    cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    best_val_err = float("inf")  # Initialize best validation error

    if cp_g and cp_do:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        encoder.load_state_dict(state_dict_g['encoder'])
        quantizer.load_state_dict(state_dict_g['quantizer'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        mstftd.load_state_dict(state_dict_do['mstftd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
    else:
        last_epoch = -1

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        encoder = DistributedDataParallel(encoder, device_ids=[rank]).to(device)
        quantizer = DistributedDataParallel(quantizer, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
        mstftd = DistributedDataParallel(mstftd, device_ids=[rank]).to(device)

    optim_g = torch.optim.Adam(
        itertools.chain(generator.parameters(), encoder.parameters(), quantizer.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.Adam(
        itertools.chain(msd.parameters(), mpd.parameters(), mstftd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])

    if cp_do:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)
    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels, h.hop_size,
                          h.win_size, h.sampling_rate, h.fmin, h.fmax, shuffle=h.num_gpus == 1,
                          device=device, fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    train_loader = DataLoader(trainset, num_workers=h.num_workers, sampler=train_sampler,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels, h.hop_size,
                              h.win_size, h.sampling_rate, h.fmin, h.fmax, device=device,
                              fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, batch_size=1, pin_memory=True, drop_last=True)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    encoder.train()
    quantizer.train()
    mpd.train()
    msd.train()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        for i, batch in enumerate(train_loader):
            steps += 1

            # Validation
            if steps % a.validation_interval == 0 and steps != 0:
                generator.eval()
                encoder.eval()
                quantizer.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0

                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, y, _, y_mel = batch
                        c = encoder(y.to(device).unsqueeze(1))
                        q, loss_q, c = quantizer(c)
                        y_g_hat = generator(q)
                        y_mel = torch.autograd.Variable(y_mel.to(device))
                        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                      h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                        i_size = min(y_mel.size(2), y_g_hat_mel.size(2))
                        val_err_tot += F.l1_loss(y_mel[:, :, :i_size], y_g_hat_mel[:, :, :i_size]).item()

                    val_err = val_err_tot / (j + 1)
                    sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    if val_err < best_val_err:
                        best_val_err = val_err
                        save_best_checkpoint(a.checkpoint_path, {
                            'generator': generator.state_dict(),
                            'encoder': encoder.state_dict(),
                            'quantizer': quantizer.state_dict()
                        }, "g")

                        save_best_checkpoint(a.checkpoint_path, {
                            'mpd': mpd.state_dict(),
                            'msd': msd.state_dict(),
                            'mstftd': mstftd.state_dict(),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'steps': steps,
                            'epoch': epoch
                        }, "do")

                generator.train()

        scheduler_g.step()
        scheduler_d.step()
