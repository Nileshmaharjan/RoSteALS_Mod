
import os, sys, torch
import argparse
from pathlib import Path
import numpy as np
from torchvision import transforms
import argparse
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from tools.eval_metrics import compute_psnr, compute_ssim, compute_mse, compute_lpips, compute_sifid
import lpips
from tools.sifid import SIFID
from tools.helpers import welcome_message
from tools.ecc import ECC


def main(args):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(welcome_message())
    # Load model
    config = OmegaConf.load(args.config).model
    secret_len = config.params.control_config.params.secret_len
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

    # cover
    tform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    cover_org = Image.open(args.cover).convert('RGB')
    cover_org = cover_org.resize((256, 256))
    # w,h = cover_org.size
    w= 256
    h = 256
    cover = tform(cover_org).unsqueeze(0).cuda()  # 1, 3, 256, 256

    # secret
    ecc = ECC()
    secret = ecc.encode_text([args.secret])  # 1, 100
    secret = torch.from_numpy(secret).cuda().float()  # 1, 100


    # inference

    with torch.no_grad():
        # z = model.encode_first_stage(cover)
        # z_embed, _ = model(z, None, secret)
        #
        # stego = model.decode_first_stage(z_embed)  # 1, 3, 256, 256
        # res = stego.clamp(-1,1) - cover  # (1,3,256,256) residual
        # res = torch.nn.functional.interpolate(res, (h,w), mode='bilinear')
        # res = res.permute(0,2,3,1).cpu().numpy()  # (1,h,w,3)
        #
        # stego_uint8 = np.clip(res[0] + np.array(cover_org)/127.5-1., -1,1)*127.5+127.5
        # stego_uint8 = stego_uint8.astype(np.uint8)  # (h,w, 3), ndarray, uint8


        # Image.fromarray(stego_uint8).save(args.output)
        # print(f'Stego saved to {args.output}')

        embedded_image = Image.open(args.e)
        embedded_image_np = np.array(embedded_image).astype(np.uint8)
        print('here')

        # are_equal = np.array_equal(stego_uint8, embedded_image_np)
        # print('here')

        # decode secret
        print('Extracting secret...')
        # Decode from stego_uint8
        #stego_tensor = torch.from_numpy(stego_uint8).permute(2, 0, 1).unsqueeze(0).cuda().float() / 127.5 - 1.
        stego_tensor = torch.from_numpy(embedded_image_np).permute(2, 0, 1).unsqueeze(0).cuda().float() / 127.5 - 1.
        decoded_secret = model.decoder(stego_tensor)
        secret_pred = (decoded_secret > 0).cpu().numpy()  # 1, 100
        print(f'Bit acc: {np.mean(secret_pred == secret.cpu().numpy())}')
        secret_decoded = ecc.decode_text(secret_pred)[0]
        print(f'Recovered secret: {secret_decoded}')

        # # save stego
        # Image.fromarray(stego_uint8).save(args.output)
        # print(f'Stego saved to {args.output}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", default='models/VQ4_s100_mir100k2.yaml', help="Path to config file.")
    parser.add_argument('-w', "--weight", default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/controlnet/VQ4_s100_mir100k2/checkpoints/epoch=000017-step=000449999.ckpt', help="Path to checkpoint file.")
    parser.add_argument(
        "--image_size", type=int, default=256, help="Height and width of square images."
    )
    parser.add_argument(
        "--secret", default='abcdef', help="secret message, 7 characters max"
    )
    parser.add_argument(
        "--cover", default='examples/00096.png', help="cover image path"
    )
    parser.add_argument(
        "-o", "--output", default='stego.png', help="output stego image path"
    )
    parser.add_argument(
        "--e", default='stego.png', help="embedded stego image path"
    )
    args = parser.parse_args()
    main(args)