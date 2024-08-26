import random
import time

import PIL
import torch
import numpy as np, copy, os, sys
import matplotlib.pyplot as plt

from utils import * # image save utils

from stable_diffusion import load_stable_diffusion, encode_latent, decode_latent, get_text_embedding, get_unet_layers, attention_op
from config.run_args import prompts, sty_fns
import copy

# For visualizing features
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torchvision.transforms as T
import cv2
from tqdm import tqdm




parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config.config import get_args
import warnings


warnings.filterwarnings("ignore")
# class for obtain and override the features
class style_transfer_module():
           
    def __init__(self,
        unet, vae, text_encoder, tokenizer, scheduler, prompt_encoder, args, style_transfer_params = None
    ):  
        
        style_transfer_params_default = {
            'gamma': 0.75,
            'tau': 1.5,
            'injection_layers': [7, 8, 9, 10, 11]
        }
        if style_transfer_params is not None:
            style_transfer_params_default.update(style_transfer_params)
        self.style_transfer_params = style_transfer_params_default
        
        self.unet = unet # SD unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.prompt_encoder = prompt_encoder

        self.attn_features = {} # where to save key value (attention block feature)
        self.attn_features_modify = {} # where to save key value to modify (attention block feature)

        self.cur_t = None
        self.thr_timestep = args.timestep_thr  
        self.guidance_scale = 7.5    
        self.args = args  
        self.hook_handles = {}
        # Get residual and attention block in decoder
        # [0 ~ 11], total 12 layers
        _ , attn, _ = get_unet_layers(unet)
        
        # where to inject key and value
        qkv_injection_layer_num = self.style_transfer_params['injection_layers']
    
        
        for i in qkv_injection_layer_num:
            self.attn_features["layer{}_attn".format(i)] = {}
            handle = attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__get_query_key_value("layer{}_attn".format(i)))
            self.hook_handles[f"get_qkv_{i}"] = handle  # Save handle

        
        # Modify hook (if you change query key value)
        for i in qkv_injection_layer_num:
            handle = attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__modify_self_attn_qkv(name="layer{}_attn".format(i), thr_timestep=self.thr_timestep))
            self.hook_handles[f"modify_qkv_{i}"] = handle  # Save handle
        # triggers for obtaining or modifying features
        
        self.trigger_get_qkv = False # if set True --> save attn qkv in self.attn_features
        self.trigger_modify_qkv = False # if set True --> save attn qkv by self.attn_features_modify
        
        self.modify_num = None # ignore
        self.modify_num_sa = None # ignore
        
    def remove_hooks(self):
        for handle in self.hook_handles.values():
            handle.remove()  # Remove the hook
        self.hook_handles.clear()  # Clear the dictionary
    
    def get_text_condition(self, text):

        if text is None:
            text = ""
            text_embeddings, uncond_embeddings = get_text_embedding(text, self.prompt_encoder)
            denoise_kwargs = {
                'encoder_hidden_states': torch.cat((uncond_embeddings, uncond_embeddings), dim=0)
            }
            return denoise_kwargs

        text_embeddings, uncond_embeddings = get_text_embedding(text, self.prompt_encoder)
        # print(f"Text embeddings shape {text_embeddings.shape} and Uncond Embeddings shape {uncond_embeddings.shape}")
        denoise_kwargs = {
            'encoder_hidden_states': torch.cat((text_embeddings, uncond_embeddings), dim=0)
        }
        return denoise_kwargs
    
  
    
    def reverse_process(self, input, denoise_kwargs):
        pred_images = []
        pred_latents = []
        
        decode_kwargs = {'vae': self.vae}

     
        # print(f"Input reverse shape {input.shape}")
        
        # Reverse diffusion process
        for t in tqdm(self.scheduler.timesteps):
            
            # setting t (for saving time step)
            self.cur_t = t.item()
            
            #Exponentially increase the injection percentage
            # self.style_transfer_params['gamma'] *= 1.0585

            
            with torch.no_grad():
                 # For text condition on stable diffusion
                if 'encoder_hidden_states' in denoise_kwargs.keys():
                    # if cfg.prompt is not None:
                    bs = denoise_kwargs['encoder_hidden_states'].shape[0]
                    # print(f"Batch size reverse {bs}")
                    input = torch.cat([input] * bs)
                
               
                # print(f"Current time step {t} with input shape {input.shape} and denoise_kwargs shape {denoise_kwargs['encoder_hidden_states'].shape}")
                noisy_residual = self.unet(input, t.to(input.device), **denoise_kwargs).sample
                    
                # For text condition on stable diffusion
                if noisy_residual.shape[0] == 2:
                    # perform guidance
                    noise_pred_text, noise_pred_uncond = noisy_residual.chunk(2)
                    noisy_residual = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    input, _ = input.chunk(2)

                prev_noisy_sample = self.scheduler.step(noisy_residual, t, input).prev_sample                # coef * P_t(e_t(x_t)) + D_t(e_t(x_t))
                pred_original_sample = self.scheduler.step(noisy_residual, t, input).pred_original_sample    # D_t(e_t(x_t))
                
                input = prev_noisy_sample
                
                pred_latents.append(pred_original_sample)
                pred_images.append(decode_latent(pred_original_sample, **decode_kwargs))
                
        return pred_images, pred_latents
        
     ## Inversion (https://github.com/huggingface/diffusion-models-class/blob/main/unit4/01_ddim_inversion.ipynb)
    def invert_process(self, input,denoise_kwargs):

        pred_images = []
        pred_latents = []
        
        decode_kwargs = {'vae': self.vae}

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.scheduler.timesteps)
        num_inference_steps = len(self.scheduler.timesteps)

        with torch.no_grad():
            # For text condition on stable diffusion
            if 'encoder_hidden_states' in denoise_kwargs.keys():
                bs = denoise_kwargs['encoder_hidden_states'].shape[0]
                # print(f"Batch size invert {bs}")
                input = torch.cat([input] * bs)
                # print(f"Input invert shape {input.shape}")

            for i in tqdm(range(0, num_inference_steps)):

                t = timesteps[i]
                
                self.cur_t = t.item()
                
             

                # Predict the noise residual
                noisy_residual = self.unet(input, t.to(input.device), **denoise_kwargs).sample
                
                noise_pred = noisy_residual

                # For text condition on stable diffusion
                if noisy_residual.shape[0] == 2:
                    # perform guidance
                    noise_pred_text, noise_pred_uncond = noisy_residual.chunk(2)
                    noisy_residual = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    input, _ = input.chunk(2)

                current_t = max(0, t.item() - (1000//num_inference_steps)) #t
                next_t = t # min(999, t.item() + (1000//num_inference_steps)) # t+1
                alpha_t = self.scheduler.alphas_cumprod[current_t]
                alpha_t_next = self.scheduler.alphas_cumprod[next_t]

                latents = input
                # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
                latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
                
                input = latents
                
                pred_latents.append(latents)
                pred_images.append(decode_latent(latents, **decode_kwargs))
                
        return pred_images, pred_latents
    




    # ============================ hook operations ===============================
    
    # save key value in self.original_kv[name]
    def __get_query_key_value(self, name):
        def hook(model, input, output):
            
            if self.trigger_get_qkv:
                    
                _, query, key, value, _ = attention_op(model, input[0])
                
                self.attn_features[name][int(self.cur_t)] = (query.detach(), key.detach(), value.detach())
            
        return hook

    
    def __modify_self_attn_qkv(self,thr_timestep, name):
        def hook(model, input, output):
        
            if self.trigger_modify_qkv:
                module_name = name # TODO
                # print(f"Input shape[0] is {input[0].shape} while input is {input}")
              
                
                _ , q_cs, k_cs, v_cs, _ = attention_op(model, input[0],slice=False)
                # print(f"q_cs shape: {q_cs.shape} and k_cs shape: {k_cs.shape} and v_cs shape: {v_cs.shape}")
                
                _, k_s, v_s = self.attn_features_modify[name][int(self.cur_t)]

               

                if self.args.start:
                    if self.cur_t > thr_timestep:
                    
                        k_cs = k_s * self.style_transfer_params['gamma'] + k_cs * (1 - self.style_transfer_params['gamma'])
                        v_cs = v_s * self.style_transfer_params['gamma'] + v_cs * (1 - self.style_transfer_params['gamma'])

                else:
                    if self.cur_t <= thr_timestep:
                        
                        k_cs = k_s * self.style_transfer_params['gamma'] + k_cs * (1 - self.style_transfer_params['gamma'])
                        v_cs = v_s * self.style_transfer_params['gamma'] + v_cs * (1 - self.style_transfer_params['gamma'])
                        

                

                
                _, _, _, _, modified_output = attention_op(model, input[0], key=k_cs, value=v_cs, query=q_cs, temperature=self.style_transfer_params['tau'])
                
                return modified_output
        
        return hook
    
    
if __name__ == "__main__":

    torch.cuda.empty_cache()
    args = get_args().parse_args()

    # options
    ddim_steps = args.ddim_steps
    device = "cuda"
    dtype = torch.float16
    in_c = 4
    
    
    style_transfer_params = {
        'gamma': args.gamma,
        'tau': args.T,
        'injection_layers': args.layers,
    }
    # self.config = config
    device = "cuda" if torch.cuda.is_available() else "cpu"

        
    
    # Get SD modules
    vae, tokenizer, text_encoder, unet, scheduler,prompt_encoder,pipe = load_stable_diffusion(sd_version=str(args.sd_version), precision_t=dtype)
    scheduler.set_timesteps(args.ddim_steps)
    sample_size = unet.config.sample_size
    


    
    for prompt in prompts:
        for sty in sty_fns:
            
            # Result save at save_dir
            save_dir = args.save_dir    
            os.makedirs(save_dir, exist_ok=True)

            style_image = cv2.imread(sty)[:, :, ::-1]
            style_image_jpg_name = sty.split("/")[-1].split(".")[0]


        
            print(f"Prompt: {prompt[0]}")
            print(f"Syle name: {style_image_jpg_name}")
            print(f"Token Indices: {prompt[1]}")

    

            # Init style transfer module
            unet_wrapper = style_transfer_module(unet, vae, text_encoder, tokenizer, scheduler, prompt_encoder,args=args, style_transfer_params=style_transfer_params)
    
    
            # Get style image tokens
            denoise_kwargs = unet_wrapper.get_text_condition(None)
    
            unet_wrapper.trigger_get_qkv = True # get attention features (key, value)
            unet_wrapper.trigger_modify_qkv = False
    
            style_latent = encode_latent(normalize(style_image).to(device=vae.device, dtype=dtype), vae)

            # invert process
            print("Invert style image...")
            images, latents = unet_wrapper.invert_process(style_latent, denoise_kwargs=denoise_kwargs) # reverse process save activations such as attn, res
            style_latent = latents[-1]
            
            images = [denormalize(input)[0] for input in images]
            image_last = images[-1]
            images = np.concatenate(images, axis=1)
            

            
            # ================= IMPORTANT =================
            # save key value from style image
            style_features = copy.deepcopy(unet_wrapper.attn_features)
            # =============================================

            
            
            
            # ================= IMPORTANT =================
            # Set modify features
            for layer_name in style_features.keys():
                unet_wrapper.attn_features_modify[layer_name] = {}
                for t in scheduler.timesteps:
                    t = t.item()
                    unet_wrapper.attn_features_modify[layer_name][t] = (style_features[layer_name][t][0], style_features[layer_name][t][1], style_features[layer_name][t][2]) # content as q / style as kv   

            # =============================================
 
                
            unet_wrapper.trigger_get_qkv = False # get attention features (key, value)
            unet_wrapper.trigger_modify_qkv = False

            if args.initno:
            
                if args.without_init_adain:
                    print("With initno style transfer...")
                    print("Without init adain style transfer...")
                    torch.manual_seed(args.seed)
                    generator = torch.Generator(device).manual_seed(args.seed)

                    latent_cs_rand = torch.randn((1,4,64,64), device=device).half()
                    

                    latent_pipe = pipe(
                        prompt=prompt[0],
                        token_indices=prompt[1],
                        guidance_scale=7.5,
                        generator=generator,
                        num_inference_steps=args.ddim_steps,
                        max_iter_to_alter=args.ddim_steps/2,
                        latents=latent_cs_rand,
                        result_root=None,
                        seed=args.seed,
                        run_sd=False,
                        return_dict=False,
                    )
                    
                    latent_cs = latent_pipe

                else:
                    style_latent = style_latent[:1]
                    print("With initno style transfer...")
                    print("With init adain style transfer...")

                    torch.manual_seed(args.seed)
                    generator = torch.Generator(device).manual_seed(args.seed)
                    latent_cs = torch.randn((1,4,64,64), device=device)
                    latent_cs = latent_cs.half()

                    latent_cs = (latent_cs - latent_cs.mean(dim=(2, 3), keepdim=True)) / (latent_cs.std(dim=(2, 3), keepdim=True) + 1e-4) * style_latent.std(dim=(2, 3), keepdim=True) + style_latent.mean(dim=(2, 3), keepdim=True)

                    
                    
                        
                    latent_pipe = pipe(
                        prompt=prompt[0],
                        token_indices=prompt[1],
                        guidance_scale=7.5,
                        generator=generator,
                        num_inference_steps=args.ddim_steps,
                        max_iter_to_alter=args.ddim_steps/2,
                        latents=latent_cs,
                        result_root=None,
                        seed=args.seed,
                        run_sd=False,
                        return_dict=False,
                    )
                        
                    
                    latent_cs = latent_pipe



            else:
                    torch.manual_seed(args.seed)
                    print("Without initno style transfer...")
                    print("With init adain style transfer...")


                    latent = torch.randn((1,4,64,64), device=device)
                    latent_cs = latent.half() * scheduler.init_noise_sigma
                    style_latent = style_latent[:1]
                   
                  
                    latent_cs = (latent_cs - latent_cs.mean(dim=(2, 3), keepdim=True)) / (latent_cs.std(dim=(2, 3), keepdim=True) + 1e-4) * style_latent.std(dim=(2, 3), keepdim=True) + style_latent.mean(dim=(2, 3), keepdim=True)




            # reverse process
            print("Style transfer...")
            
            unet_wrapper.trigger_get_qkv = False
            unet_wrapper.trigger_modify_qkv = not args.without_attn_injection # modify attn feature (key value)
           
            denoise_kwargs = unet_wrapper.get_text_condition(prompt[0])

            # print(f"Style Image Prompt: {style_text} and Content Image Prompt: {content_text}")
            images, latents = unet_wrapper.reverse_process(latent_cs, denoise_kwargs=denoise_kwargs) # reverse process save activations such as attn, res


            unet_wrapper.remove_hooks()

            images = [denormalize(input)[0] for input in images]
            image_last = images[-1]
            generated_image = image_last

            images = np.concatenate(images, axis=1)
            generated_seq = images


            if args.initno:
                # save_dir = os.path.join(save_dir, "initno")
                # if args.without_init_adain:
                #     save_dir = os.path.join(save_dir, "without_init_adain", style_image_jpg_name, str(args.timestep_thr))
                #     os.makedirs(save_dir, exist_ok=True)
                
                
                save_dir = os.path.join(save_dir,style_image_jpg_name, str(args.timestep_thr),str(args.seed))
                os.makedirs(save_dir, exist_ok=True)

            # else:
            #     save_dir = os.path.join(save_dir, "without_initno")
            #     if args.without_init_adain:
            #         save_dir = os.path.join(save_dir, "without_init_adain", style_image_jpg_name, str(args.timestep_thr))
            #         os.makedirs(save_dir, exist_ok=True)
            #     else:
            #         save_dir = os.path.join(save_dir, "with_init_adain",style_image_jpg_name, str(args.timestep_thr))
            #         os.makedirs(save_dir, exist_ok=True)
                


            save_image(images, os.path.join(save_dir, f"gen_seq_{prompt[0]}_{args.seed}_{args.gamma}.jpg"))
            save_image(image_last, os.path.join(save_dir, f"gen_{prompt[0]}_{args.seed}_{args.gamma}.jpg"))
            


    
