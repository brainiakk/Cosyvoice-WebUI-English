**Start web demo**

You can use our web demo page to get familiar with CosyVoice quickly.
We support sft/zero_shot/cross_lingual/instruct inference in web demo.

Please see the demo website for details.
``` bash
git clone https://github.com/brainiakk/Cosyvoice-WebUI-English.git
cd Cosyvoice-WebUI-English/CosyVoice
mkdir pretrained_models
```
**Download Models**

CosyVoice-300M HF Link: https://huggingface.co/FunAudioLLM/CosyVoice-300M <br/>
CosyVoice-300M-SFT HF Link: https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT <br/>
CosyVoice-300M-Instruct HF Link: https://huggingface.co/FunAudioLLM/CosyVoice-300M-Instruct <br/>

Make sure you have downloaded the models & copied them to the pretrained_models/Model directory

e.g:<br/>
pretrained_models/CosyVoice-300M<br/>
pretrained_models/CosyVoice-300M-SFT<br/>
pretrained_models/CosyVoice-300M-Instruct

Depending on the model you downloaded

``` python
# change iic/CosyVoice-300M-SFT for sft inference, or iic/CosyVoice-300M-Instruct for instruct inference
python -m webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```