# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['Pre-trained Voice', '3s Rapid Cloning', 'Cross-lingual Cloning', 'Natural Language Control']
instruct_dict = {
    'Pre-trained Voice': '1. Select pre-trained voice\n2. Click generate audio button',
    '3s Rapid Cloning': '1. Select or record prompt audio (max 30s). File upload takes priority if both provided\n2. Enter prompt text\n3. Click generate audio button',
    'Cross-lingual Cloning': '1. Select or record prompt audio (max 30s). File upload takes priority if both provided\n2. Click generate audio button',
    'Natural Language Control': '1. Select pre-trained voice\n2. Enter instruct text\n3. Click generate audio button'
}
stream_mode_list = [('No', False), ('Yes', True)]
max_val = 0.8

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    if mode_checkbox_group in ['Natural Language Control']:
        if cosyvoice.frontend.instruct is False:
            gr.Warning(f'You are using Natural Language Control mode. {args.model_dir} model does not support this mode. Please use iic/CosyVoice-300M-Instruct model')
            yield (target_sr, default_data)
        if instruct_text == '':
            gr.Warning('You are using Natural Language Control mode. Please enter instruct text')
            yield (target_sr, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('You are using Natural Language Control mode. Prompt audio/prompt text will be ignored')

    if mode_checkbox_group in ['Cross-lingual Cloning']:
        if cosyvoice.frontend.instruct is True:
            gr.Warning(f'You are using Cross-lingual Cloning mode. {args.model_dir} model does not support this mode. Please use iic/CosyVoice-300M model')
            yield (target_sr, default_data)
        if instruct_text != '':
            gr.Info('You are using Cross-lingual Cloning mode. Instruct text will be ignored')
        if prompt_wav is None:
            gr.Warning('You are using Cross-lingual Cloning mode. Please provide prompt audio')
            yield (target_sr, default_data)
        gr.Info('You are using Cross-lingual Cloning mode. Please ensure synthesis text and prompt text are in different languages')

    if mode_checkbox_group in ['3s Rapid Cloning', 'Cross-lingual Cloning']:
        if prompt_wav is None:
            gr.Warning('Prompt audio is empty. Did you forget to input prompt audio?')
            yield (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning(f'Prompt audio sample rate {torchaudio.info(prompt_wav).sample_rate} is lower than {prompt_sr}')
            yield (target_sr, default_data)

    if mode_checkbox_group in ['Pre-trained Voice']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('You are using Pre-trained Voice mode. Prompt text/prompt audio/instruct text will be ignored!')

    if mode_checkbox_group in ['3s Rapid Cloning']:
        if prompt_text == '':
            gr.Warning('Prompt text is empty. Did you forget to input prompt text?')
            yield (target_sr, default_data)
        if instruct_text != '':
            gr.Info('You are using 3s Rapid Cloning mode. Pre-trained voice/instruct text will be ignored!')

    if mode_checkbox_group == 'Pre-trained Voice':
        logging.info('Received SFT inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3s Rapid Cloning':
        logging.info('Received zero-shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'Cross-lingual Cloning':
        logging.info('Received cross-lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    else:
        logging.info('Received instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Repository [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    Pre-trained Models [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### Please enter the text to synthesize, select the inference mode, and follow the prompts")

        tts_text = gr.Textbox(label="Enter synthesis text", lines=1, value="I am a generative speech large model newly launched by the speech team of Tongyi Lab, providing comfortable and natural speech synthesis capabilities.")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Select inference mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Operation steps", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Select pre-trained voice', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='Stream inference?', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed adjustment (non-streaming only)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Random inference seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Select prompt audio file, sample rate not less than 16kHz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record prompt audio file')
        prompt_text = gr.Textbox(label="Enter prompt text", lines=1, placeholder="Please enter prompt text, it should match the prompt audio content. Auto-recognition is not supported currently...", value='')
        instruct_text = gr.Textbox(label="Enter instruct text", lines=1, placeholder="Please enter instruct text.", value='')

        generate_button = gr.Button("Generate Audio")

        audio_output = gr.Audio(label="Synthesized Audio", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()
