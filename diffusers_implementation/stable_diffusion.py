
import torch
# from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from initno.pipelines.pipeline_sd_initno import StableDiffusionInitNOPipeline
from diffusers import LMSDiscreteScheduler, DDIMScheduler
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config.config import get_args
import warnings


warnings.filterwarnings("ignore")


args=get_args().parse_args()


# From "https://huggingface.co/blog/stable_diffusion"
def load_stable_diffusion(sd_version='2.1', precision_t=torch.float32, device="cuda"):
    if sd_version == '2.1':
        model_key = "stabilityai/stable-diffusion-2-1-base"
    elif sd_version == '2.0':
        model_key = "stabilityai/stable-diffusion-2-base"
    elif sd_version == '1.5':
        model_key = "runwayml/stable-diffusion-v1-5"
  


    if args.initno:
        # Create model
        
        
        pipe = StableDiffusionInitNOPipeline.from_pretrained(model_key,torch_dtype=precision_t).to(device)

        SEED = args.seed
        # PROMPT          = args.prompt
        # token_indices   = args.token_indices

        # print(f"From get_indices: {pipe.get_indices(PROMPT)}")

      
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t).to(device)
        latent = None
        

    
        
    vae = pipe.vae
    # get_indices = pipe.get_indices
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    prompt_encoder = pipe.encode_prompt
    
    
    unet = pipe.unet
    
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    # call_fn.to(device)
    # print(f"Latent shape: {latent.shape} and latent type: {type(latent)}")
    
    
    
    
    # del pipe
    

    # Use DDIM scheduler
    scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=precision_t)
    
    return vae, tokenizer, text_encoder, unet, scheduler, prompt_encoder, pipe
# , mask_image

def decode_latent(latents, vae):
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    return image

def encode_latent(images, vae):
    # encode the image with vae
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mode()
    latents = 0.18215 * latents
    return latents

def get_text_embedding(text, prompt_encoder , device="cuda"):
    # TODO currently, hard-coding for stable diffusion
    with torch.no_grad():
        return prompt_encoder(text,device,1,True)



def get_unet_layers(unet):
    layer_num = [i for i in range(12)]
    resnet_layers = []
    attn_layers = []
    feature_maps = {}

    def hook(module, input, output):
        # print(f"Module in hook: {module}")
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            feature_maps[module] = output

    for idx, ln in enumerate(layer_num):
        up_block_idx = idx // 3
        layer_idx = idx % 3

        resnet_layer = getattr(unet, 'up_blocks')[up_block_idx].resnets[layer_idx]
        resnet_layers.append(resnet_layer)

        # Register the hook to the Conv2d sub-modules
        for sub_module in resnet_layer.modules():
            if isinstance(sub_module, torch.nn.modules.conv.Conv2d):
                sub_module.register_forward_hook(hook)

        if up_block_idx > 0:
            attn_layer = getattr(unet, 'up_blocks')[up_block_idx].attentions[layer_idx]
            attn_layers.append(attn_layer)

            # Register the hook to the Conv2d sub-modules
            for sub_module in attn_layer.modules():
                if isinstance(sub_module, torch.nn.modules.conv.Conv2d):
                    sub_module.register_forward_hook(hook)
        else:
            attn_layers.append(None)

    return resnet_layers, attn_layers, feature_maps
        
        

# Diffusers attention code for getting query, key, value and attention map
def attention_op(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query=None, key=None, value=None, attention_probs=None, temperature=1.0,slice=False):
    # print("Hidden states shape in attention_op: ", hidden_states.shape)
    

    if slice:
        uncond_hidden_states = hidden_states[-1:]
        hidden_states = hidden_states[:1]
      
        # print("Hidden states shape after slicing: ", hidden_states.shape)

    residual = hidden_states
    
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    if query is None:
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    if key is None:
        key = attn.to_k(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
    if value is None:
        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)

    
    if key.shape[0] != query.shape[0]:
        key, value = key[:query.shape[0]], value[:query.shape[0]]

    # apply temperature scaling
    query = query * temperature # same as applying it on qk matrix

    if attention_probs is None:
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        

    batch_heads, img_len, txt_len = attention_probs.shape
    
    # h = w = int(img_len ** 0.5)
    # attention_probs_return = attention_probs.reshape(batch_heads // attn.heads, attn.heads, h, w, txt_len)
    
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor
    
    return attention_probs, query, key, value, hidden_states


