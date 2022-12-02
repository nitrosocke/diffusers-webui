import gradio as gr
import random
import os
import torch
import time
import sys
import numpy as np
from PIL import Image, PngImagePlugin
from configparser import ConfigParser
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import (DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler)

print(f"Starting Demo...")

start_time = time.time()
save_set_txt = False
vmodel = True

class Model:
    def __init__(self, name, path="", prefix=""):
        self.name = name
        self.path = path
        self.prefix = prefix
        self.pipe_t2i = None
        self.pipe_i2i = None
  
models = []

model_paths = [f.path for f in os.scandir("./models") if f.is_dir()]
model_names = [f.name for f in os.scandir("./models") if f.is_dir()]
for i in range(len(model_paths)):
  models.insert(i, Model(f"{model_names[int(i)]}",f"{model_paths[int(i)]}"))

#Get the configparser object
config_object = ConfigParser()

config_object.read("config.ini")
userinfo = config_object["USERINFO"]
imgc = int(userinfo["last_img"])
print("Saving images to: {}".format(userinfo["save_path"]))

last_mode = "txt2img"

if not models:
  print("")
  print("No models found! Please place your models into the 'models' folder")
  sys.exit()
  
current_model = models[0]
current_model_path = None
current_save_path = userinfo["save_path"]
gsettings = ""
scheduler_set = ""

#current_scheduler= EulerDiscreteScheduler.from_pretrained(current_model.path, subfolder="scheduler", prediction_type="v_prediction")
current_scheduler= None
scheduler_set = None


# Load model
#print(f"Loading model {current_model.name}...")
#pipe = StableDiffusionPipeline.from_pretrained(current_model_path, torch_dtype=torch.float16, scheduler=current_scheduler)

#uncomment the line below to enable xformers
#pipe.enable_xformers_memory_efficient_attention()

def error_str(error, title="Error"):
    return f"""#### {title}
            {error}"""  if error else ""

def custom_model_changed(path):
  models[0].path = path
  global current_model
  current_model = models[0]

def on_vmodel_change(set_vmodel):
  global vmodel
  vmodel = set_vmodel
  if vmodel == True:
    print("Loading models with v-prediction")
  else:
    print("Loading models with eps-prediction")

def on_scheduler_change(n_scheduler, vmodel):
  if n_scheduler == "euler":
    if vmodel == True:
      new_scheduler = EulerDiscreteScheduler.from_pretrained(current_model.path, subfolder="scheduler", prediction_type="v_prediction")
    else: new_scheduler = EulerDiscreteScheduler.from_pretrained(current_model.path, subfolder="scheduler", prediction_type="epsilon")
  elif n_scheduler == "dpm++":
    if vmodel == True:
      new_scheduler = DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler", prediction_type="v_prediction")
    else: new_scheduler = DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler", prediction_type="epsilon")
  elif n_scheduler == "dpm":
    if vmodel == True:
      new_scheduler = DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler", prediction_type="v_prediction", algorithm_type="dpmsolver")
    else: new_scheduler = DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler", algorithm_type="dpmsolver", prediction_type="epsilon")
  elif n_scheduler == "heun":
    if vmodel == True:
      new_scheduler = DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler", prediction_type="v_prediction", solver_type="heun")
    else: new_scheduler = DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler", solver_type="heun", prediction_type="epsilon")
  elif n_scheduler == "ddpm":
    if vmodel == True:
      new_scheduler = DDPMScheduler.from_pretrained(current_model.path, subfolder="scheduler", prediction_type="v_prediction")
    else: new_scheduler = DDPMScheduler.from_pretrained(current_model.path, subfolder="scheduler", prediction_type="epsilon")
  elif n_scheduler == "lms":
    new_scheduler = LMSDiscreteScheduler.from_pretrained(current_model.path, subfolder="scheduler")
  elif n_scheduler == "pndm":
    new_scheduler = PNDMScheduler.from_pretrained(current_model.path, subfolder="scheduler")
  elif n_scheduler == "ddim":
    new_scheduler = DDIMScheduler.from_pretrained(current_model.path, subfolder="scheduler")
  else:
    if vmodel == True:
      new_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(current_model.path, subfolder="scheduler", prediction_type="v_prediction")
    else: new_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(current_model.path, subfolder="scheduler", prediction_type="epsilon")
  global scheduler_set 
  scheduler_set = n_scheduler

  global current_scheduler
  current_scheduler = new_scheduler
  global pipe
  pipe = StableDiffusionPipeline.from_pretrained(current_model_path, torch_dtype=torch.float16, scheduler=current_scheduler)
  pipe = pipe.to("cuda")

def save_path_changed(spath):
  img_save_path = str(spath)
  global current_save_path
  current_save_path = img_save_path
  userinfo["save_path"] = f"{current_save_path}"
  #Write changes back to file
  with open('config.ini', 'w') as conf:
    config_object.write(conf)
  print(f"Saving images to {current_save_path}")

def settings_output(results, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height):
  global gsettings
  gsettings = (f"Prompt: {prompt} Negative Prompt: {negative_prompt}, Scheduler: {scheduler_set}, Steps: {num_inference_steps}, CFG: {guidance_scale}, {width}x{height}, Seed: {gen_seed} Model: {current_model.name}")
  print(f"{gsettings}")

  update_settings(gsettings)
  return save_images(results)

def update_settings(gsettings2):
  settings = gsettings2
  return(settings)


def inference(model_name, prompt, guidance, steps, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt="", n_images=1, scheduler="euler"):

  global current_model
  global gen_seed
  global vmodel

  for model in models:
    if model.name == model_name:
      current_model = model
      model_path = current_model.path

  if seed != 0:
    generator = torch.Generator('cuda').manual_seed(seed)
    gen_seed = seed
  else:
    rnd_seed = random.randint(0, 2147483647)
    generator = torch.Generator('cuda').manual_seed(rnd_seed)
    gen_seed = rnd_seed

  try:
    os.mkdir (current_save_path)
  except OSError as error:
    print("")
  try:
    if img is not None:
      return img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator), None, gsettings
    else:
      return txt_to_img(model_path, prompt, neg_prompt, n_images, guidance, steps, width, height, generator, scheduler), None, gsettings
  except Exception as e:
    return None, error_str(e), None
  

def txt_to_img(model_path, prompt, neg_prompt, n_images, guidance, steps, width, height, generator, scheduler):

    global current_scheduler
    global last_mode
    global pipe
    global current_model_path

    if model_path != current_model_path or last_mode != "txt2img":
        current_model_path = model_path
        on_scheduler_change(scheduler, vmodel)
        pipe = StableDiffusionPipeline.from_pretrained(current_model_path, torch_dtype=torch.float16, scheduler=current_scheduler)        

        pipe = pipe.to("cuda")
        last_mode = "txt2img"

    if scheduler_set != scheduler:
      on_scheduler_change(scheduler, vmodel)

    print(f"txt_to_img, model: {current_model.name} with {scheduler_set}")

    prompt = f"{prompt}"  
    results = pipe(
      prompt,
      negative_prompt = neg_prompt,
      num_images_per_prompt=n_images,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator)

    return settings_output(results, prompt, neg_prompt, int(steps), guidance, width, height)

def img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator):

    global last_mode
    global pipe
    global current_model_path

    if model_path != current_model_path or last_mode != "img2img":
        current_model_path = model_path

        pipe = pipe.to("cuda")
        last_mode = "img2img"

    prompt = f"{prompt}"
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    print(f"img_to_img, model: {model_path} with {scheduler_set}")
    
    results = pipe(
        prompt,
        negative_prompt = neg_prompt,
        num_images_per_prompt=n_images,
        init_image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        width = width,
        height = height,
        generator = generator)
    
    return save_images(results)

def save_images(results):
  global imgc
  rnd = random.randint(100000,9999999)
  info = PngImagePlugin.PngInfo()
  info.add_text("text", f"{gsettings}")
  info.add_text("ZIP", "VALUE", zip=True)
  for img in results.images:
    
    img.save(current_save_path+"/"+f'{imgc:04d}'+"-"+str(rnd)+".png", "PNG", pnginfo=info)
    imgc=imgc+1

  userinfo["last_img"] = str(imgc)
  #Write changes back to file
  with open('config.ini', 'w') as conf:
    config_object.write(conf)

  return results.images

css = """.finetuned-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.finetuned-diffusion-div div h1{font-weight:900;margin-bottom:7px}.finetuned-diffusion-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""
with gr.Blocks(css=css) as demo:
    gr.HTML(
        f"""
            <div class="diffusers-webui-div">
              <div>
                <h1>Diffusers WebUI</h1>
              </div>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Column(scale=55):
          with gr.Group():
              gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery"
              ).style(grid=[2], height="auto", container=True)

          error_output = gr.Markdown()

        with gr.Column(scale=45):
          with gr.Tab("Options"):
            generate = gr.Button(value="Generate", variant="secondary").style(container=False)
            with gr.Group():
              prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=3,placeholder="Enter prompt", lines=3).style(container=False)
              
              neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")

              n_images = gr.Slider(label="Images", value=1, minimum=1, maximum=12, step=1)

              with gr.Row():
                guidance = gr.Slider(label="Guidance scale", value=7, maximum=15, step=1)
                steps = gr.Slider(label="Steps", value=20, minimum=2, maximum=50, step=1)

              with gr.Row():
                width = gr.Slider(label="Width", value=768, minimum=64, maximum=1920, step=64)
                height = gr.Slider(label="Height", value=768, minimum=64, maximum=1920, step=64)

              scheduler_dd = gr.Radio(label="Scheduler", choices=["euler_a", "euler", "dpm++", "ddim", "ddpm", "pndm", "lms", "heun", "dpm"], value="euler", type="value")
              seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)
              settings = gr.Markdown()
              

          with gr.Tab("Image to image"):
              with gr.Group():
                image = gr.Image(label="Image", height=256, tool="editor", type="pil")
                strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)

          with gr.Tab("Settings"):
              with gr.Group():
                model_name = gr.Dropdown(label="Model", choices=[m.name for m in models], value=current_model.name)
                save_vmodel = gr.Checkbox(label="V-Model", value=True)

              with gr.Group():
                save_path = gr.Textbox(label="Image save Path", value=current_save_path)
                save_settings_button= gr.Button(value="Save Path").style(container=True)
    allsettings = [save_path]
    save_vmodel.change(on_vmodel_change,inputs=save_vmodel, outputs=None)

    inputs = [model_name, prompt, guidance, steps, width, height, seed, image, strength, neg_prompt, n_images, scheduler_dd]
    outputs = [gallery, error_output, settings]
    prompt.submit(inference, inputs=inputs, outputs=outputs, show_progress=True)
    generate.click(inference, inputs=inputs, outputs=outputs, show_progress=True)
    save_settings_button.click(save_path_changed, inputs=allsettings, outputs=error_output)

    ex = gr.Examples([
       ["beautiful female siren mystical human creature", "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy", 7, 20],
       ["the heroes journey, a never ending labyrinth, epic emotional fantasy art by tomasz alen kopera, artgerm, peter mohrbacher, donato giancola, joseph christian leyendecker, wlop, boris vallejo", "blurry, bad art, bad anatomy, blurred, text, watermark, grainy", 7, 20],
       ["beautiful fjord at sunrise", "fog blurry soft tiling bad art grainy", 7, 20],
      
    ], inputs=[prompt, neg_prompt, guidance, steps, seed], outputs=outputs, fn=inference, cache_examples=False)

    gr.HTML("""
    <div style="border-top: 1px solid #303030;">
      <br>
      <p>Made by Nitrosocke.</p>
    </div>
    """)

print(f"Started in {time.time() - start_time:.2f} seconds")

demo.launch()
