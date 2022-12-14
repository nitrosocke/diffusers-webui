import gradio as gr, re
import random
import os
import glob
import torch
import time
import sys
import numpy as np
from PIL import Image, PngImagePlugin
from configparser import ConfigParser
from transformers import pipeline, set_seed
from lora_diffusion import monkeypatch_lora, tune_lora_scale
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DiffusionPipeline
from diffusers import (DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler)

print(f"Starting Demo...")

start_time = time.time()
save_set_txt = False
vmodel = True
use_xformers = True
gtp2_load = False
gtp2 = ""
i2i_mode = False
use_emb = False
emb_chng = True
cemb_name = None
te_name = None

class Model:
    def __init__(self, name, path="", prefix=""):
        self.name = name
        self.path = path
        self.prefix = prefix
        self.pipe_t2i = None
        self.pipe_i2i = None

class Embedding:
    def __init__(self, name):
        self.name = name
        self.pipe_t2i = None
        self.pipe_i2i = None
  
models = []
embeddings = []

model_paths = [f.path for f in os.scandir("./models") if f.is_dir()]
model_names = [f.name for f in os.scandir("./models") if f.is_dir()]
emb_names = [f.name for f in os.scandir("./embeddings") if f.is_file()]
for i in range(len(model_paths)):
  models.insert(i, Model(f"{model_names[int(i)]}",f"{model_paths[int(i)]}"))

emb_names[:] = [x for x in emb_names if not x.endswith("encoder.pt")]

if range(len(emb_names)) != 0:
  for i in range(len(emb_names)):
    embeddings.insert(i, Embedding(f"{emb_names[int(i)]}"))

#Get the configparser object
config_object = ConfigParser()

config_object.read("config.ini")
userinfo = config_object["USERINFO"]
imgc = int(userinfo["last_img"])
lmodel = int(userinfo["last_model"])
use_xformers = bool(userinfo["enable_xformers"])
print(f"Using Xformers: {use_xformers}")
print("Saving images to: {}".format(userinfo["save_path"]))

last_mode = "txt2img"

if not models:
  print("No models found! Please place your models into the 'models' folder")
  sys.exit()

if not embeddings:
  print("No embeddings found!")
  embeddings.insert(0, Embedding(f"No Embeddings"))
  current_emb = embeddings[0]
else: 
  current_emb = embeddings[0]
  print(f"{len(emb_names)} embeddings found")

current_model = models[lmodel]
current_model_path = None
current_save_path = userinfo["save_path"]
gsettings = ""
scheduler_set = ""

current_scheduler= None
scheduler_set = None

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
    return("Loading models with v-prediction")
  else:
    print("Loading models with eps-prediction")
    return("Loading models with eps-prediction")

def on_xformers_change(set_xformers):
  global use_xformers
  use_xformers = set_xformers
  
  userinfo["enable_xformers"] = str(use_xformers)
  #Write changes back to file
  with open('config.ini', 'w') as conf:
    config_object.write(conf)

  if use_xformers == True:
    print("Xformers enabled, please restart the webUI")
  else:
    print("Xformers disabled, please restart the webUI")
  return("Xformers setting changed, please restart the webUI")

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
  global i2i_mode
  global emb_chng

  if i2i_mode == False:
    pipe = StableDiffusionPipeline.from_pretrained(current_model_path, custom_pipeline="lpw_stable_diffusion", torch_dtype=torch.float16, scheduler=current_scheduler)
  else: 
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(current_model_path, custom_pipeline="lpw_stable_diffusion", torch_dtype=torch.float16, scheduler=current_scheduler)
  pipe = pipe.to("cuda")
  if use_xformers == True:
    pipe.enable_xformers_memory_efficient_attention()
  else:
    pipe.disable_xformers_memory_efficient_attention()
  emb_chng = True

def refresh_path():
  global models
  global embeddings
  global current_emb
  global current_model
  global current_model_path
  models = []
  embeddings = []

  model_paths = [f.path for f in os.scandir("./models") if f.is_dir()]
  model_names = [f.name for f in os.scandir("./models") if f.is_dir()]
  emb_names = [f.name for f in os.scandir("./embeddings") if f.is_file()]
  for i in range(len(model_paths)):
    models.insert(i, Model(f"{model_names[int(i)]}",f"{model_paths[int(i)]}"))

  emb_names[:] = [x for x in emb_names if not x.endswith("encoder.pt")]

  if range(len(emb_names)) != 0:
    for i in range(len(emb_names)):
      embeddings.insert(i, Embedding(f"{emb_names[int(i)]}"))
 
  print("Updating Directories")
  return (gr.Dropdown.update(choices=[e.name for e in embeddings], value=current_emb.name),gr.Dropdown.update(choices=[m.name for m in models], value=current_model.name), f"Refreshed Models: {len(model_paths)} and Embeddings: {len(emb_names)}")


def save_path_changed(spath):
  img_save_path = str(spath)
  global current_save_path
  current_save_path = img_save_path
  userinfo["save_path"] = f"{current_save_path}"
  #Write changes back to file
  with open('config.ini', 'w') as conf:
    config_object.write(conf)
  print(f"Saving images to {current_save_path}")
  return(f"Saving images to {current_save_path}")

def settings_output(results, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, embname, emb_weight, te_weight):
  global gsettings
  global use_emb
  if use_emb == True:
    gsettings = (f"Prompt: {prompt} Negative Prompt: {negative_prompt}, Scheduler: {scheduler_set}, Steps: {num_inference_steps}, CFG: {guidance_scale}, {width}x{height}, Seed: {gen_seed}, Model: {current_model.name}, Embedding: {embname}, Weights: Emb {emb_weight}, Te {emb_weight}")
  else:
    gsettings = (f"Prompt: {prompt} Negative Prompt: {negative_prompt}, Scheduler: {scheduler_set}, Steps: {num_inference_steps}, CFG: {guidance_scale}, {width}x{height}, Seed: {gen_seed}, Model: {current_model.name}")
  print(f"{gsettings}")

  update_settings(gsettings)
  return save_images(results)

def update_settings(gsettings2):
  settings = gsettings2
  return(settings)

def on_use_emb(value):
  global use_emb
  global last_mode
  if value == True:
    use_emb = True
    print("Using Embeddings")
  else:
    use_emb = False
    print("Not using Embeddings")
    last_mode = "emb_chng"

def on_emb_change(embname):
  global emb_chng
  global current_emb
  global last_mode
  global cemb_name
  global te_name
  if use_emb == True:
    last_mode = "emb_chng"
  emb_chng = True
  cemb_name = embname
  te_name = cemb_name.split(".")
  print(f"changing embedding {cemb_name}")

def inference(model_name, prompt, guidance, steps, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt="", n_images=1, scheduler="euler_a", emb_weight=1.0, te_weight=1.0):

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
      return img_to_img(model_path, prompt, neg_prompt, img, strength, n_images, guidance, steps, width, height, generator, scheduler, emb_weight, te_weight), None, gsettings
    else:
      return txt_to_img(model_path, prompt, neg_prompt, n_images, guidance, steps, width, height, generator, scheduler, emb_weight, te_weight), None, gsettings
  except Exception as e:
    return None, error_str(e), None
  

def txt_to_img(model_path, prompt, neg_prompt, n_images, guidance, steps, width, height, generator, scheduler, emb_weight, te_weight):

    global current_scheduler
    global last_mode
    global pipe
    global current_model_path
    global i2i_mode
    global scheduler_set
    global emb_chng
    global cemb_name
    global te_name

    if model_path != current_model_path or last_mode != "txt2img":
      print("loading model...")
      i2i_mode = False
      current_model_path = model_path
      on_scheduler_change(scheduler, vmodel)
      last_mode = "txt2img"

    if scheduler_set != scheduler:
      print("switching scheduler...")
      i2i_mode = False
      on_scheduler_change(scheduler, vmodel)

    if cemb_name == None:
      cemb_name = current_emb.name
      te_name = cemb_name.split(".")

    if use_emb == True and emb_chng == True:
      print(f"Using embedding: {cemb_name}")
      monkeypatch_lora(pipe.unet, torch.load(f"embeddings/{cemb_name}"))
      try:
        monkeypatch_lora(pipe.text_encoder, torch.load(f"embeddings/{te_name[0]}.text_encoder.pt"), target_replace_module=["CLIPAttention"])
      except:
        print("no text encoder found")
      emb_chng = False

    if use_emb == True:
      tune_lora_scale(pipe.unet, emb_weight)
      tune_lora_scale(pipe.text_encoder, te_weight)

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
      generator = generator,
      max_embeddings_multiples = 3)

    return settings_output(results, prompt, neg_prompt, int(steps), guidance, width, height, cemb_name, emb_weight, te_weight)

def img_to_img(model_path, prompt, neg_prompt, img, strength, n_images, guidance, steps, width, height, generator, scheduler, emb_weight, te_weight):

    global current_scheduler
    global last_mode
    global pipe
    global current_model_path
    global i2i_mode
    global scheduler_set
    global emb_chng
    global cemb_name
    global te_name

    if model_path != current_model_path or last_mode != "img2img":
      print("loading model...")
      i2i_mode = True
      current_model_path = model_path
      on_scheduler_change(scheduler, vmodel)
      last_mode = "img2img"

    if scheduler_set != scheduler:
      print("switching scheduler...")
      i2i_mode = True
      on_scheduler_change(scheduler, vmodel)

    if cemb_name == None:
      cemb_name = current_emb.name
      te_name = f"{current_emb}.text_encoder.pt"
      print(te_name)

    if use_emb == True and emb_chng == True:
      print(f"Using embedding: {cemb_name}")
      monkeypatch_lora(pipe.unet, torch.load(f"embeddings/{cemb_name}"))
      try:
        monkeypatch_lora(pipe.text_encoder, torch.load(f"embeddings/{te_name[0]}.text_encoder.pt"), target_replace_module=["CLIPAttention"])
      except:
        print("no text encoder found")
      emb_chng = False

    if use_emb == True:
      tune_lora_scale(pipe.unet, emb_weight)
      tune_lora_scale(pipe.text_encoder, te_weight)

    print(f"img_to_img, model: {current_model.name} with {scheduler_set}")

    prompt = f"{prompt}"
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)

    print("Image resized")
    
    results = pipe(
        prompt,
        negative_prompt = neg_prompt,
        num_images_per_prompt=n_images,
        image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        #width = width,
        #height = height,
        generator = generator)
    
    return settings_output(results, prompt, neg_prompt, int(steps), guidance, width, height, cemb_name, emb_weight)

def save_images(results):
  global imgc
  rnd = random.randint(100000,9999999)
  info = PngImagePlugin.PngInfo()
  info.add_text("parameters", f"{gsettings}")
  info.add_text("ZIP", "VALUE", zip=True)
  for img in results.images:
    
    img.save(current_save_path+"/"+f'{imgc:05d}'+"-"+str(rnd)+".png", "PNG", pnginfo=info)
    imgc=imgc+1

  userinfo["last_img"] = str(imgc)
  userinfo["last_model"] = str(int(models.index(current_model)))
  #Write changes back to file
  with open('config.ini', 'w') as conf:
    config_object.write(conf)

  return results.images

def on_png_info(image):
  if image != None:
    print("Getting PNG info")
    #print(image.info)
    print(image.info["parameters"])
    pngText = image.info["parameters"]
    return(pngText)

def generate_prompt(starting_text):
  try:
    gpt2_pipe = pipeline('text-generation', model='tools/GPT2', tokenizer='gpt2')
  except:
    print("No model found, please place the GPT2 model inside the 'tools' folder")
    return

  seed = random.randint(100, 1000000)
  set_seed(seed)

  response = gpt2_pipe(starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=1)
  response_list = []
  for x in response:
    resp = x['generated_text'].strip()
    if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
      response_list.append(resp+'\n')

  response_end = "\n".join(response_list)
  response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
  response_end = response_end.replace("<", "").replace(">", "")

  if response_end != "":
    global gtp2
    gtp2 = response_end
    return response_end

def gtp2_prompt():
  return(gtp2)

css = """.finetuned-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.finetuned-diffusion-div div h1{font-weight:900;margin-bottom:7px}.finetuned-diffusion-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""
with gr.Blocks(css=css) as demo:

    gr.HTML(
        f"""
            <div class="diffusion-spave-div">
              <div>
                <h1>Diffusion Space</h1>
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
          
          settings = gr.Markdown()
          error_output = gr.Markdown()

        with gr.Column(scale=45):
          with gr.Tab("Main"):
            generate = gr.Button(value="Generate", variant="secondary").style(container=False)
            with gr.Group():
              prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=3,placeholder="Enter prompt", lines=3).style(container=False)
              
              neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")

              with gr.Row():
                n_images = gr.Slider(label="Images", value=1, minimum=1, maximum=12, step=1)
                seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)

              with gr.Row():
                guidance = gr.Slider(label="Guidance scale", value=7, maximum=20, step=1)
                steps = gr.Slider(label="Steps", value=20, minimum=2, maximum=50, step=1)

              with gr.Row():
                width = gr.Slider(label="Width", value=768, minimum=64, maximum=1920, step=64)
                height = gr.Slider(label="Height", value=768, minimum=64, maximum=1920, step=64)

              scheduler_dd = gr.Radio(label="Scheduler", choices=["euler_a", "euler", "dpm++", "ddim", "ddpm", "pndm", "lms", "heun", "dpm"], value="euler_a", type="value")

            with gr.Group(): 
              with gr.Row():  
                use_emb = gr.Checkbox(label="Use Embedding", value=False, ).style(container=True)
                emb_name = gr.Dropdown(label="Emb Name", choices=[e.name for e in embeddings], value=current_emb.name, interactive=True, show_label=False)
              with gr.Row():
                emb_weight = gr.Slider(label="EMB Weight", minimum=0, maximum=1, step=0.05, value=1)
                te_weight = gr.Slider(label="TE Weight", minimum=0, maximum=1, step=0.05, value=1)

          with gr.Tab("Img2Img"):
            with gr.Group():
              image = gr.Image(label="Image", height=256, tool="editor", type="pil")
              strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)
            generate2 = gr.Button(value="Generate", variant="secondary").style(container=False)

          with gr.Tab("Tools"):
            with gr.Group():
              pngInfo = gr.Image(label="PNG Chunk Explorer", height=256, type="pil")
              pngText = gr.Markdown()

            with gr.Group():
              txt = gr.Textbox(lines=1, label="Initial Text", placeholder="Prompt start here", show_label=False)
              out = gr.Textbox(lines=4, label="Generated Prompts")
              gen = gr.Button(value="Make Prompt").style(container=False)
              send_prompt = gr.Button(value="Send to Prompt").style(container=False)
            gr.HTML("""
              <div style="border-top: 1px solid #303030;">
                <br>
                <p>Using MagicPrompt by <a href="https://huggingface.co/Gustavosta">Gustavosta</a> <3</p>
              </div>
              """)

          with gr.Tab("Settings"):
            refresh = gr.Button(value="Refresh Models").style(container=True)
            settings_message = gr.Markdown()
            with gr.Group():
              model_name = gr.Dropdown(label="Model", choices=[m.name for m in models], value=current_model.name)
              save_vmodel = gr.Checkbox(label="V-Model", value=True)
              save_xformers = gr.Checkbox(label="Xformers", value=use_xformers)

            with gr.Group():
              save_path = gr.Textbox(label="Image save Path", value=current_save_path)
              save_path_button= gr.Button(value="Save Path").style(container=True)

    allsettings = [save_path]
    pngInfo.change(on_png_info,inputs=pngInfo,outputs=pngText)
    use_emb.change(on_use_emb,inputs=use_emb,outputs=None)
    save_vmodel.change(on_vmodel_change,inputs=save_vmodel, outputs=settings_message)
    save_xformers.change(on_xformers_change,inputs=save_xformers, outputs=settings_message)
    emb_name.change(on_emb_change, inputs=emb_name, outputs=None)

    inputs = [model_name, prompt, guidance, steps, width, height, seed, image, strength, neg_prompt, n_images, scheduler_dd, emb_weight, te_weight]
    outputs = [gallery, error_output, settings]
    refresh_out = [emb_name, model_name, settings_message]
    prompt.submit(inference, inputs=inputs, outputs=outputs, show_progress=True)
    generate.click(inference, inputs=inputs, outputs=outputs, show_progress=True)
    generate2.click(inference, inputs=inputs, outputs=outputs, show_progress=True)
    gen.click(generate_prompt, inputs=txt, outputs=out, show_progress=True)
    send_prompt.click(gtp2_prompt, inputs=None, outputs=prompt, scroll_to_output=True)
    save_path_button.click(save_path_changed, inputs=allsettings, outputs=settings_message)
    refresh.click(refresh_path, inputs=None, outputs=refresh_out)

    exp = gr.Examples([
      ["beautiful female witch"],
      ["beautiful portrait of a girl with demon horns and blonde hair, by Ilya Kuvshinov"],
      ["centered, profile picture, simple, barbie, bright, beautiful girl, teenager, blonde hair, feminine, shimmering, sparkle, girlie, pink, princesss"],
      ["cute cotton candy girl, realistic, photo real, fantasy, pastel colors, rainbow curve hair"],
      ["orange and black, head shot of a woman standing under street lights, dark theme, Frank Miller, cinema, ultra realistic, ambiance, insanely detailed and intricate, hyper realistic, 8k resolution, photorealistic, highly textured, intricate details"],
      ["beautiful fjord at sunrise"],
      ], inputs=[prompt], outputs=outputs, fn=inference, cache_examples=False, label="Prompts")

    exnp = gr.Examples([
      ["ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"],
      ["blurry, bad art, bad anatomy, blurred, text, watermark, grainy"],
      ["ugly, deformed, disfigured, malformed, blurry, mutated, extra limbs, bad anatomy, cropped, floating limbs, disconnected limbs"],
      ["blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, double, blurred, disfigured, deformed, repetitive, black and white"],
      ["bad art, strange colours, sketch, lacklustre, repetitive, cropped, lowres, deformed, old, childish"],
      ["blender, text, disfigured, realistic, photo, 3d render, fused fingers, malformed"],
      ["blender, text, disfigured, realistic, photo, 3d render, grain, cropped, out of frame"],
      ["fog blurry soft tiling bad art grainy"],
      ], inputs=[neg_prompt], outputs=outputs, fn=inference, cache_examples=False, label="Negative Prompts")

    set = gr.Examples([
          [512,512],
          [768,1024],
          [1024, 768],
          [1920, 1088],
      ], inputs=[width, height], outputs=outputs, fn=inference, cache_examples=False, label="Resolutions")

    gr.HTML("""
    <div style="border-top: 1px solid #303030;">
      <br>
      <p>WebUI made by Nitrosocke.</p>
    </div>
    """)

print(f"Space built in {time.time() - start_time:.2f} seconds")

demo.launch()
