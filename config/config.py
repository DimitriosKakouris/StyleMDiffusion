import argparse
import random

def get_args():
    parser = argparse.ArgumentParser()

    # prompts = ['a cat', 'a dog', 'a car','an airplane', 'a portrait of a man', 'a portrait of a woman']


    # Arguments from config.py
    parser.add_argument('--T', type=int, default=1.5)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--without_init_adain', action='store_true')
    parser.add_argument('--without_attn_injection', type=bool, default=False)
    parser.add_argument('--layers', nargs='+', type=int, default=[7, 8, 9, 10, 11])
    parser.add_argument('--ddim_steps', type=int, default=20)
    parser.add_argument('--sd_version', type=float, choices=[1.4, 1.5, 2.1], default=2.1)
    parser.add_argument('--cnt_fn', type=str, required=False)
    parser.add_argument('--sty_fn', type=str, required=False)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--start',type=int,default=0)
    # parser.add_argument('--style_prompt', type=str, default='a field of flowers')
    # parser.add_argument('--test', action='store_false')
    parser.add_argument('--timestep_thr', type=int, default=40)
    # parser.add_argument('--model_path', default='models/EfficientNet_B2_FT.pth', type=str)
    # parser.add_argument('--ckpt', default='models/StyleID.pth', type=str)
    # parser.add_argument('--train', action='store_true')
    # parser.add_argument('--prompt', type=str, required=True)
    # parser.add_argument('--num_epochs', type=int, default=1)
    # parser.add_argument('--batch_size', type=int, default=10)
    # parser.add_argument('--token_indices', nargs='+', type=int, default=[1])
    parser.add_argument('--initno', action='store_true')

    # Arguments from cfg.py
    # parser.add_argument('--mode', default='infer', type=str, help='Mode (infer |train | test)')
    # parser.add_argument('--model', default='effnetb2', type=str, help='Model to use' )
    # parser.add_argument('--style_img', default='/StyleMergeDiffusion/StyleID/data_vis/sty/the_starry_night.png', help='Style Image to classify')
    # parser.add_argument('--train_dataset', default='/kaggle/working/input/', help='Path to training dataset')

    return parser