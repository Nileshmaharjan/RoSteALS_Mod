#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Note: there are some hard-coded fields in this script, so it wont work without modification. This file serves as an illustration for the evaluation process of RoSteALS only.

same as eval_imagec.py but for all perturbations on every image on CLIC, MetFace and Stock
ideally on small dataset
@author: Tu Bui @University of Surrey
"""

import os, sys, torch 
import argparse
from pathlib import Path
import numpy as np
import pickle
import pytorch_lightning as pl
from torchvision import transforms
import argparse
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from time import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tools.eval_metrics import compute_psnr, compute_ssim, compute_mse, compute_lpips, compute_sifid, resize_array, resize_tensor
from tools.augment_imagenetc import RandomImagenetC
import lpips
from tools.sifid import SIFID
from tools.image_dataset import dataset_wrapper
from tools.helpers import welcome_message
import bchlib
# BCH_POLYNOMIAL = 137
# BCH_BITS = 5

def unormalize(x):
    # convert x in range [-1, 1], (B,C,H,W), tensor to [0, 255], uint8, numpy, (B,H,W,C)
    x = torch.clamp((x + 1) * 127.5, 0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return x


class ECC(object):
    def __init__(self, BCH_POLYNOMIAL = 137, BCH_BITS = 5):
        self.bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    
    def _generate(self):
        dlen = 56
        data= torch.zeros(dlen, dtype=torch.float).random_(0, 2).numpy()
        data_str = ''.join(str(x) for x in data.astype(int))
        packet = bytes(int(data_str[i: i + 8], 2) for i in range(0, dlen, 8))
        packet = bytearray(packet)
        ecc = self.bch.encode(packet)
        packet = packet + ecc  # 96 bits
        packet = ''.join(format(x, '08b') for x in packet)
        packet = [int(x) for x in packet]
        packet.extend([0, 0, 0, 0])
        packet = np.array(packet, dtype=np.float32)  # 100
        return packet, data

    def generate(self, nsamples=1):
        # generate random 56 bit secret
        data = [self._generate() for _ in range(nsamples)]
        data = (np.array([d[0] for d in data]), np.array([d[1] for d in data]))
        return data  # data with ecc, data org
    
    def _decode(self, x):
        packet_binary = "".join([str(int(bit)) for bit in x])
        packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        data, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
        bitflips = self.bch.decode_inplace(data, ecc)
        if bitflips == -1:  # error, return wrong data
            data = np.ones(56, dtype=np.float32)*2. 
        else:
            data = ''.join(format(x, '08b') for x in data)
            data = np.array([int(x) for x in data], dtype=np.float32)
        return data  # 56 bits

    def decode(self, data):
        """Decode secret with BCH ECC and convert to string.
        Input: secret (torch.tensor) with shape (B, 100) type bool
        Output: secret (B, 56)"""
        data = data[:, :96]
        data = [self._decode(d) for d in data]
        return np.array(data)


def identity(x):
    return x 

def main(args):
    print(welcome_message())
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.resize_before_metric:
        print('Resize before computing metric')
        resize_array_fn = resize_array
        resize_tensor_fn = resize_tensor
    else:
        print('Use designed resolution for metric')
        resize_array_fn = identity
        resize_tensor_fn = identity

    # Load model
    config = OmegaConf.load(args.config).model
    secret_len = config.params.control_config.params.secret_len
    if args.ecc:
        assert secret_len == 100, 'ECC only support 100 bits secret (for now)'
    config.params.decoder_config.params.secret_len = secret_len
    model = instantiate_from_config(config)
    state_dict = torch.load(args.weight, map_location=torch.device('cpu'))
    if 'global_step' in state_dict:
        print(f'Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    misses, ignores = model.load_state_dict(state_dict, strict=False)
    print(f'Missed keys: {misses}\nIgnore keys: {ignores}')
    model = model.cuda()
    model.eval()

    # test
    tform = transforms.Compose([
        # transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    lpips_alex = lpips.LPIPS(net='alex').cuda()
    sifid_model = SIFID()
    noise = RandomImagenetC(1, 5, 'test')
    noise_ids = noise.corrupt_ids
    noise_strengths = np.array([1,2,3,4,5]) 

    if args.ecc:
        ecc = ECC()

    dataset_all = [
        ('/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/data/clic','/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/data/clic/clic.csv'),
        ('/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/data/metface/images', '/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/data/metface/metface.csv'),
        ('/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/data/stock1k/Stock_Watermark_Test', '/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/data/stock1k/stock1k.csv')
    ]
    for data_dir, data_list in dataset_all:
        dname = data_list.split('/')[-1].split('.')[0]
        print(f'Processing {dname}...')
        # Load image
        dataset = dataset_wrapper(data_dir, data_list, secret_len=secret_len, transform=transforms.Resize((args.image_size, args.image_size)))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(dataset)

        score_lpips, score_sifid, score_ssim, score_psnr, score_mse = [], [], [], [], []  # quality of stego vs original cover
        score_lpips_ae, score_sifid_ae, score_ssim_ae, score_psnr_ae, score_mse_ae = [], [], [], [], []  # quality of AE reconstruction
        score_lpips_recon, score_sifid_recon, score_ssim_recon, score_psnr_recon, score_mse_recon = [], [], [], [], []  # quality of stego vs recon

        bit_acc = {i: [] for i in noise_ids}
        bit_acc[-1] = []  # -1 for clean stego
        # print(bit_acc.keys())
        noise_level = {i: [] for i in noise_ids}  # mirror bit acc but store noise levels for later analysis
        if args.ecc:
            bit_ecc = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                img, secret = batch['image'].cuda(), batch['secret'].cuda()
                img = img.permute(0, 3, 1, 2)  # B, 3, 256, 256 in range [-1, 1]
                img_z = model.encode_first_stage(img)  # (B, 4, 64, 64)
                recon_ae = model.decode_first_stage(img_z)  # (B, 3, 256, 256) in range [-1, 1], recon using AE only, no stego
                z, _ = model(img_z, None, secret)  # (B, 4, 64, 64)
                stego = model.decode_first_stage(z)  # (B, 3, 256, 256) in range [-1, 1]

                # convert to np image array (B, H, 256, 256) in range [0, 255]
                img_unorm = resize_array_fn(unormalize(img))
                stego_unorm = resize_array_fn(unormalize(stego))
                recon_ae_unorm = resize_array_fn(unormalize(recon_ae))

                # eval stego quality: SSIM, PSNR, MSE, LPIPS
                score_lpips.append(compute_lpips(resize_tensor_fn(img), resize_tensor_fn(stego), lpips_alex))
                score_sifid.append(compute_sifid(resize_tensor_fn(img), resize_tensor_fn(stego), sifid_model))
                score_ssim.append(compute_ssim(img_unorm, stego_unorm))
                score_psnr.append(compute_psnr(img_unorm, stego_unorm))
                score_mse.append(compute_mse(img_unorm, stego_unorm))

                # eval ae quality: SSIM, PSNR, MSE, LPIPS
                score_lpips_ae.append(compute_lpips(resize_tensor_fn(img), resize_tensor_fn(recon_ae), lpips_alex))
                score_sifid_ae.append(compute_sifid(resize_tensor_fn(img), resize_tensor_fn(recon_ae), sifid_model))
                score_ssim_ae.append(compute_ssim(img_unorm, recon_ae_unorm))
                score_psnr_ae.append(compute_psnr(img_unorm, recon_ae_unorm))
                score_mse_ae.append(compute_mse(img_unorm, recon_ae_unorm))

                # eval stego quality vs recon: SSIM, PSNR, MSE, LPIPS
                score_lpips_recon.append(compute_lpips(resize_tensor_fn(recon_ae), resize_tensor_fn(stego), lpips_alex))
                score_sifid_recon.append(compute_sifid(resize_tensor_fn(recon_ae), resize_tensor_fn(stego), sifid_model))
                score_ssim_recon.append(compute_ssim(recon_ae_unorm, stego_unorm))
                score_psnr_recon.append(compute_psnr(recon_ae_unorm, stego_unorm))
                score_mse_recon.append(compute_mse(recon_ae_unorm, stego_unorm))

                secret = secret.cpu().numpy()
                secret_pred = (model.decoder(stego) > 0).cpu().numpy()
                bit_acc[-1].append(np.mean(secret == secret_pred, axis=1))
                # perturb stego
                for noise_id in noise_ids:
                    levels = np.random.choice(noise_strengths, len(stego_unorm))
                    stegos_perturbed = [tform(noise(Image.fromarray(im), noise_id, level)) for im, level in zip(stego_unorm, levels)]
                    stegos_perturbed = torch.stack(stegos_perturbed).cuda()

                    # predict secret perturbed
                    secret_pred = (model.decoder(stegos_perturbed) > 0).cpu().numpy()
                    bit_acc[noise_id].append(np.mean(secret == secret_pred, axis=1))
                    noise_level[noise_id].append(levels)

                # ecc
                if args.ecc:
                    secret_ecc, secret_org = ecc.generate(img_z.shape[0])
                    secret_ecc = torch.from_numpy(secret_ecc).cuda()
                    z, _ = model(img_z, None, secret_ecc)  # (B, 4, 64, 64)
                    stego = model.decode_first_stage(z)  # (B, 3, 256, 256) in range [-1, 1]
                    stego_unorm = resize_array_fn(unormalize(stego))
                    noise_id = np.random.choice(noise_ids, len(stego_unorm))
                    levels = np.random.choice(noise_strengths, len(stego_unorm))
                    stegos_perturbed = [tform(noise(Image.fromarray(im), nid, level)) for im, nid, level in zip(stego_unorm, noise_id, levels)]
                    stegos_perturbed = torch.stack(stegos_perturbed).cuda()
                    secret_pred = (model.decoder(stego) > 0).cpu().numpy()
                    secret_pred = ecc.decode(secret_pred)
                    bit_ecc.append(np.mean(secret_org == secret_pred, axis=1))
        
        score_lpips, score_sifid, score_ssim, score_psnr, score_mse = [np.concatenate(x) for x in [score_lpips, score_sifid, score_ssim, score_psnr, score_mse]]

        if args.ecc:
            bit_ecc = np.concatenate(bit_ecc)

        score_lpips_ae, score_sifid_ae, score_ssim_ae, score_psnr_ae, score_mse_ae = [np.concatenate(x) for x in [score_lpips_ae, score_sifid_ae, score_ssim_ae, score_psnr_ae, score_mse_ae]]

        score_lpips_recon, score_sifid_recon, score_ssim_recon, score_psnr_recon, score_mse_recon = [np.concatenate(x) for x in [score_lpips_recon, score_sifid_recon, score_ssim_recon, score_psnr_recon, score_mse_recon]]

        bit_acc = {i: np.concatenate(x) for i, x in bit_acc.items()}
        noise_level = {i: np.concatenate(x) for i, x in noise_level.items()}

        out = {}

        print(f"mse AE: {score_mse_ae.mean():.2f}+-{score_mse_ae.std():.2f}")
        print(f"psnr AE: {score_psnr_ae.mean():.2f}+-{score_psnr_ae.std():.2f}")
        print(f"ssim AE: {score_ssim_ae.mean():.2f}+-{score_ssim_ae.std():.2f}")
        print(f"lpips AE: {score_lpips_ae.mean():.2f}+-{score_lpips_ae.std():.2f}")
        print(f"sifid AE: {score_sifid_ae.mean():.2f}+-{score_sifid_ae.std():.2f}")

        print(f"mse stego vs recon: {score_mse_recon.mean():.2f}+-{score_mse_recon.std():.2f}")
        print(f"psnr stego vs recon: {score_psnr_recon.mean():.2f}+-{score_psnr_recon.std():.2f}")
        print(f"ssim stego vs recon: {score_ssim_recon.mean():.2f}+-{score_ssim_recon.std():.2f}")
        print(f"lpips stego vs recon: {score_lpips_recon.mean():.2f}+-{score_lpips_recon.std():.2f}")
        print(f"sifid stego vs recon: {score_sifid_recon.mean():.2f}+-{score_sifid_recon.std():.2f}")

        print(f"mse: {score_mse.mean():.2f}+-{score_mse.std():.2f}")
        print(f"psnr: {score_psnr.mean():.2f}+-{score_psnr.std():.2f}")
        print(f"ssim: {score_ssim.mean():.2f}+-{score_ssim.std():.2f}")
        print(f"lpips: {score_lpips.mean():.2f}+-{score_lpips.std():.2f}")
        print(f"sifid: {score_sifid.mean():.2f}+-{score_sifid.std():.2f}")
        out.update(
            # mse=f"{score_mse.mean():.2f}+-{score_mse.std():.2f}",
                psnr=f"{score_psnr.mean():.2f}+-{score_psnr.std():.2f}",
                    ssim=f"{score_ssim.mean():.2f}+-{score_ssim.std():.2f}",
                    lpips=f"{score_lpips.mean():.2f}+-{score_lpips.std():.2f}",
                    sifid=f"{score_sifid.mean():.2f}+-{score_sifid.std():.2f}"
                    )

        for i in bit_acc:
            name = 'clean' if i==-1 else noise.method_names[i]
            # print(f"bit_acc {name}: {bit_acc[i].mean():.2f}+-{bit_acc[i].std():.2f}")
            if i==-1:
                print(f"bit_acc {name}: {bit_acc[i].mean():.3f}+-{bit_acc[i].std():.3f}")
                out.update(bit_acc_clean=f"{bit_acc[i].mean():.3f}+-{bit_acc[i].std():.3f}")


        bit_acc_noise = np.concatenate([val for i, val in bit_acc.items() if i!=-1])
        print(f"bit_acc n5: {bit_acc_noise.mean():.2f}+-{bit_acc_noise.std():.2f}")
        out.update(bit_acc=f"{bit_acc_noise.mean():.3f}+-{bit_acc_noise.std():.3f}")

        if args.ecc:
            print(f'bit acc (ecc): {bit_ecc.mean():.3f}')
            out.update(ecc=f"{bit_ecc.mean():.3f}+-{bit_ecc.std():.3f}")

        sample_n5 = (bit_acc_noise > 0.8).mean()
        print(f"word acc (t=0.2): {sample_n5:.3f}")
        out.update(sample_n5=f"{sample_n5:.3f}")

        # print all in a row
        print(' & '.join([f"{k}" for k in out.keys()]))
        print(' & '.join([f"{out[k]}" for k in out.keys()]))

        # save raw data
        raw = dict(score_lpips=score_lpips, score_sifid=score_sifid, score_ssim=score_ssim, score_psnr=score_psnr, score_mse=score_mse,

                score_lpips_ae=score_lpips_ae, score_sifid_ae=score_sifid_ae, score_ssim_ae=score_ssim_ae, score_psnr_ae=score_psnr_ae, score_mse_ae=score_mse_ae,

                score_lpips_recon=score_lpips_recon, score_sifid_recon=score_sifid_recon, score_ssim_recon=score_ssim_recon, score_psnr_recon=score_psnr_recon, score_mse_recon=score_mse_recon,

                bit_acc=bit_acc, noise_level=noise_level, noise_names=noise.method_names)
        if args.ecc:
            raw.update(bit_ecc=bit_ecc)
        with open(os.path.join(args.output, f'{dname}.pkl'), 'wb') as f:
            pickle.dump(raw, f)
        
    # # read them again with:
    # raw = pickle.load(open(args.output, 'rb'))
    # for key, val in raw.items():
    #     globals()[key] = val



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", default='models/VQ4_s100_mir100k2.yaml', help="Path to config file.")
    parser.add_argument('-w', "--weight", default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/controlnet/VQ4_s100_mir100k2/checkpoints/epoch=000017-step=000449999.ckpt', help="Path to checkpoint file.")

    parser.add_argument('-o', 
        "--output", default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/controlnet/VQ4_s100_mir100k2/ep17', help="output directory."
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Height and width of square images."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to sample fingerprints.")
    parser.add_argument("--ecc", type=int, default=0, help="perform ecc?")
    parser.add_argument('--resize_before_metric', action='store_true', help='resize image to 256x256 before computing quality metrics')
    args = parser.parse_args()

    main(args)