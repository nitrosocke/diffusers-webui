# diffusers-webui
This is a Gradio WebUI working with the Diffusers format of Stable Diffusion

![Diffusers-WebUI](https://github.com/nitrosocke/diffusers-webui/raw/main/data/images/webui-interface-preview.png)

## Windows Installation:
1. Install the recommended version of [Python 3.10.6](https://www.python.org/downloads/windows/), checking "Add Python to PATH"
2. Install [git](https://git-scm.com/download/win).
3. Download the diffusers-webui repository, for example by running `git clone https://github.com/nitrosocke/diffusers-webui.git`.
4. Place your model Diffusers in the `models` directory.
5. Run `LaunchUI.bat` from Windows Explorer as non-administrator user.

You can set an alternative Python path by editing the `LaunchUI.bat` and add the absolute path after the `set PYTHON=` like so:

`set PYTHON=C:/Python/python.exe`

## Update 1:
This update includes tons of fixes, improvements and a few feature updates:
- added MagicPrompt by Gustavosta
- added PNG Chunks writing and reading
- added img2img pipeline
- added LoRA embeddings support by cloneofsimo
- added prompt weighting support
- updated preset prompts, negative prompts and resolution
- updated UI with better tabs and labels
- fixed texts and labels

## Loading new models
The diffusers of you models folder should contain these subfolders and the 'model_index.json' file.

![Models Folder](https://github.com/nitrosocke/diffusers-webui/raw/main/data/images/models-diffuser-folder.png)

## Usage:
To start the webUI run the `LaunchUI.bat` from the directory. You can make a shortcut for it on your Desktop for easier access.
By default the webUI starts with settings optimised for the 768-v models.
You can set the model to use, the v-model option and the save path for images in the settings tab.

![Settings Tab](https://github.com/nitrosocke/diffusers-webui/raw/main/data/images/webui-settings-tab.png)

## Img2Img
Input your image you want to use and tune the 'Transformation strength' to get more of the style or more of the image. Settings from the 'Main' tab are used for generation.

## PNG Chunks
The app saves your generation settings automatically to the png files it generates. You can use the 'Tools' tab to review your settings or a website like [Project Nayuki](https://www.nayuki.io/page/png-file-chunk-inspector) to view them.

## MagicPrompt by Gustavosta
In order to use the MagicPrompt extension you need to download the model from [Gustavostas Huggingface](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion/tree/main)
You will need the files: `config.json, merges.txt, pytorch_model.bin, special_tokens_map.json, tokenizer.json, tokenizer_config.json, training_args.bin, vocab.json`
Place these files into the directory 'tools/GPT2' (the folder should be 489mb in size).

## LoRA Embeddings
You can "inject" your LoRA trained .pt files into your base model. Place your 3mb embedding file and the text encoder file (optional) into the 'embeddings' folder. For easier handling I recommend renaming the files like 'zerosuit.pt' and 'zerosuit.text_encoder.pt'. The file name has no influence on the token used for inference.
To use the embedding enable the checkbox in the 'Main' tab, select the embedding you want to use and adjust the weight slider. Make sure to prompt the trained token in your prompt. Optionally you can adjust the text encoder weight as well.

## Prompt Weighting
You can but attention to certain tokens in your prompt by using `()` or `(token:1.1)` or reduce the influence by using `[]` or `(token:0.9)`. Please see a detailed description of the usage here: [Long Prompt Weighting](https://github.com/huggingface/diffusers/tree/main/examples/community#long-prompt-weighting-stable-diffusion)

## To-Do
- implement more community pipelines
- img2img inpainting pipeline
- failsaves for missing files and dependencies
- clean up the code and comment it

## Credits:
- Stable Diffusion - [Stable-Diffusion Github](https://github.com/CompVis/stable-diffusion)
- Huggingface - [Diffusers Github](https://github.com/huggingface/diffusers), [Huggingface.co](https://huggingface.co/)
- Anzorq - for the help with the Gradio UI [Anzorq on Hugginface](https://huggingface.co/anzorq), [Finetuned-Diffusion Space](https://huggingface.co/spaces/anzorq/finetuned_diffusion)
- Gustavosta - for the amazing GPT2 MagicPrompts model: [MagicPrompt - Stable Diffusion](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion)
- cloneofsimo for the LoRA training script: [Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning](https://github.com/cloneofsimo/lora)
- SkyTNT for the [Long Prompt Weighting](https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion.py)
- Automatic1111 - for the idea and the launch script [Automatic1111 Github](https://github.com/AUTOMATIC1111)
