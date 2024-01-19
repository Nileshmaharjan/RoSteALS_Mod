
import os, sys, torch
import numpy as np
import argparse
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
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

    # secret
    ecc = ECC()
    secret = ecc.encode_text([args.secret])  # 1, 100
    secret = torch.from_numpy(secret).cuda().float()  # 1, 100


    # inference

    with torch.no_grad():

        embedded_image = Image.open(args.e)
        embedded_image_np = np.array(embedded_image).astype(np.uint8)
        print('here')

        # decode secret
        print('Extracting secret...')
        # Decode from stego_uint8

        stego_tensor = torch.from_numpy(embedded_image_np).permute(2, 0, 1).unsqueeze(0).cuda().float() / 127.5 - 1.
        decoded_secret = model.decoder(stego_tensor)
        secret_pred = (decoded_secret > 0).cpu().numpy()  # 1, 100
        print(f'Bit acc: {np.mean(secret_pred == secret.cpu().numpy())}')
        secret_decoded = ecc.decode_text(secret_pred)[0]
        print(f'Recovered secret: {secret_decoded}')


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
        "--e", default='stego.png', help="embedded stego image path"
    )
    args = parser.parse_args()
    main(args)