import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import einops
from torch.nn import functional as thf
import pytorch_lightning as pl
import torchvision
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.model import Encoder
import lpips
from kornia import color

class D2CEncoder(nn.Module):
    def __init__(self):
        super(D2CEncoder, self).__init__()
        self.secret_dense = nn.Linear(200, 3072)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.convA = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.convB = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convC = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.convD = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.convE = nn.Conv2d(160, 32, kernel_size=3, padding=1)
        self.convF = nn.Conv2d(192, 16, kernel_size=3, padding=1)


        self.conv7 = nn.Conv2d(192, 16, kernel_size=1)
        self.conv8 = nn.Conv2d(16, 16, kernel_size=1)
        self.conv9 = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, image, secret):
        secret = self.secret_dense(secret)
        secret = secret.view(secret.size(0), 3, 32, 32)
        secret_enlarged = F.interpolate(secret, scale_factor=8, mode='nearest')
        out1 = self.conv1(secret_enlarged)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        outA = self.convA(image)
        hyb_1 = torch.concat([out1, outA], dim=1)
        outB = self.convB(hyb_1)
        hyb_2 = torch.concat([out2, outA, outB], dim=1)
        outC = self.convC(hyb_2)
        hyb_3 = torch.concat([out3, outA, outB, outC], dim=1)
        outD = self.convD(hyb_3)
        hyb_4 = torch.concat([out4, outA, outB, outC, outD], dim=1)
        outE = self.convE(hyb_4)
        hyb_5 = torch.concat([out5, outA, outB, outC, outD, outE], dim=1)
        outF = self.convF(hyb_5)
        hyb_6 = torch.concat([out6, outA, outB, outC, outD, outE, outF], dim=1)
        out7 = self.conv7(hyb_6)
        out8 = self.conv8(out7)
        out9 = self.conv9(out8)
        return out9

    def encode(self, image, secret):
        encoded_image = self.forward(image, secret)

        return encoded_image


class D2CDecoder(nn.Module):
    def __init__(self, height, width):
        super(D2CDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Conv2d(3, 1, 1, padding=0),
            nn.Flatten(),
            nn.Linear(height * width, 200)
        )

    def forward(self, image):
        return self.decoder(image)

    def decode(self, image):
        decoded_data = self.forward(image)

        return decoded_data



class DenseD2C(pl.LightningModule):
    def __init__(self,
                 first_stage_key,
                 first_stage_config,
                 control_key,
                 control_config,
                 decoder_config,
                 loss_config,
                 noise_config='__none__',
                 use_ema=False,
                 secret_warmup=False,
                 scale_factor=1.,
                 ckpt_path="__none__",
                 ):
        super().__init__()

        self.control_key = control_key
        self.first_stage_key = first_stage_key
        self.ae = instantiate_from_config(first_stage_config)
        self.control = instantiate_from_config(control_config)
        self.decoder = instantiate_from_config(decoder_config)
        if noise_config != '__none__':
            print('Using noise')
            self.noise = instantiate_from_config(noise_config)


        self.fixed_x = None
        self.fixed_img = None
        self.fixed_input_recon = None
        self.fixed_control = None
        self.register_buffer("fixed_input", torch.tensor(True))



    @torch.no_grad()
    def get_input(self, batch):
        image = batch[self.first_stage_key]
        control = batch[self.control_key]

        # encode image 1st stage
        image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
        encoded_image = self.encode_first_stage(image, control).detach()

        return encoded_image



    def encode_first_stage(self, image, secret):
        encoded_image = self.ae.encode(image, secret)

        return encoded_image

    def decode_first_stage(self, encoded_image):
        decoded_data = self.decoder.decode(encoded_image)
        return decoded_data

    def shared_step(self, batch):
        encoded_image = self.get_input(batch)

        decoded_data = self.decode_first_stage(encoded_image)


        x, c, img, _ = self.get_input(batch)
        # import pdb; pdb.set_trace()
        x, posterior = self(x, img, c)
        image_rec = self.decode_first_stage(x)

        # resize
        if img.shape[-1] > 256:
            img = thf.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False).detach()
            image_rec = thf.interpolate(image_rec, size=(256, 256), mode='bilinear', align_corners=False)
        if hasattr(self, 'noise') and self.noise.is_activated():
            image_rec_noised = self.noise(image_rec, self.global_step, p=0.9)
        else:
            image_rec_noised = image_rec
        pred = self.decoder(image_rec_noised)

        loss, loss_dict = self.loss_layer(img, image_rec, posterior, c, pred, self.global_step)
        bit_acc = loss_dict["bit_acc"]

        bit_acc_ = bit_acc.item()

        if (bit_acc_ > 0.98) and (not self.fixed_input) and self.noise.is_activated():
            self.loss_layer.activate_ramp(self.global_step)

        if (bit_acc_ > 0.95) and (not self.fixed_input):  # ramp up image loss at late training stage
            if hasattr(self, 'noise') and (not self.noise.is_activated()):
                self.noise.activate(self.global_step)

        if (bit_acc_ > 0.9) and self.fixed_input:  # execute only once
            print(f'[TRAINING] High bit acc ({bit_acc_}) achieved, switch to full image dataset training.')
            self.fixed_input = ~self.fixed_input
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)


        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        loss_dict_no_ema = {f"val/{key}": val for key, val in loss_dict_no_ema.items() if key != 'img_lw'}
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {'val/' + key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    @torch.no_grad()
    def log_images(self, batch, fixed_input=False, **kwargs):
        log = dict()
        if fixed_input and self.fixed_img is not None:
            x, c, img, img_recon = self.fixed_x, self.fixed_control, self.fixed_img, self.fixed_input_recon
        else:
            x, c, img, img_recon = self.get_input(batch, return_first_stage=True)
        x, _ = self(x, img, c)
        image_out = self.decode_first_stage(x)
        if hasattr(self, 'noise') and self.noise.is_activated():
            img_noise = self.noise(image_out, self.global_step, p=1.0)
            log['noised'] = img_noise
        log['input'] = img
        log['output'] = image_out
        log['recon'] = img_recon
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)
        return optimizer