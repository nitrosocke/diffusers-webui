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

## Loading new models
The diffusers of you models folder should contain these subfolders and the 'model_index.json' file.

![Models Folder](https://github.com/nitrosocke/diffusers-webui/raw/main/data/images/models-diffuser-folder.png)

## Usage:
To start the webUI run the `LaunchUI.bat` from the directory. You can make a shortcut for it on your Desktop for easier access.
By default the webUI starts with settings optimised for the 768-v models.
You can set the model to use, the v-model option and the save path for images in the settings tab.

![Settings Tab](https://github.com/nitrosocke/diffusers-webui/raw/main/data/images/webui-settings-tab.png)

## Img2Img
This feature is not yet implemented and will be coming in a future update.

## PNG Chunks
The app saves your generation settings automatically to the png files it generates. You can use a website like [Project Nayuki](https://www.nayuki.io/page/png-file-chunk-inspector) to view them.

## Credits:
- Stable Diffusion - [Stable-Diffusion Github](https://github.com/CompVis/stable-diffusion)
- Huggingface - [Diffusers Github](https://github.com/huggingface/diffusers), [Huggingface.co](https://huggingface.co/)
- Anzorq - for his help with the Gradio UI [Anzorq on Hugginface](https://huggingface.co/anzorq), [Finetuned-Diffusion Space](https://huggingface.co/spaces/anzorq/finetuned_diffusion)
- Automatic1111 - for the idea and the launch script [Automatic1111 Github](https://github.com/AUTOMATIC1111)
