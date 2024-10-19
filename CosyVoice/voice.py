import os
import sys

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import pygame
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

class Voice:
    def __init__(self):
        self.cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
        # self.cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
        # self.cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
        # print(self.cosyvoice.list_avaliable_spks())

    def say(self, text):
        # sft usage
        filename = ""
        print(self.cosyvoice.list_avaliable_spks())
        # change stream=True for chunk stream inference
        for i, j in enumerate(self.cosyvoice.inference_sft(text, '中文男', stream=False)):
            filename = 'sft_{}.wav'.format(i)
            torchaudio.save(filename, j['tts_speech'], 22050)

        self.play(filename)
    def clone(self, text, cross_lang = ""):
        # sft usage
        filename = ""
        print(self.cosyvoice.list_avaliable_spks())
        allowed_lang = ['<|zh|>', '<|en|>', '<|jp|>', '<|yue|>', '<|ko|>']
        if cross_lang is not None and cross_lang in allowed_lang:
            text = cross_lang+text
            # cross_lingual usage
            prompt_speech_16k = load_wav('cross_lingual_prompt.wav', 16000)
            for i, j in enumerate(self.cosyvoice.inference_cross_lingual(text, prompt_speech_16k, stream=False)):
                filename = 'cross_lingual_{}.wav'.format(i)
                torchaudio.save(filename, j['tts_speech'], 22050)
        else:
            # zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
            prompt_speech_16k = load_wav('cogman-s.wav', 16000)
            for i, j in enumerate(self.cosyvoice.inference_zero_shot(text,'No, he\'s going to die. I was making the moment more epic. Leprechauns are tiny, green, and Irish, and that is offensive.', prompt_speech_16k, stream=False)):
                filename = 'zero_shot_{}.wav'.format(i)
                torchaudio.save(filename, j['tts_speech'], 22050)

        self.play(filename)

    def prompt_tts(self, text, prompt=""):
        filename = ""
        # instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
        for i, j in enumerate(self.cosyvoice.inference_instruct(text,'中文男', prompt, stream=False)):
            filename = 'instruct_{}.wav'.format(i)
            torchaudio.save(filename, j['tts_speech'], 22050)

        self.play(filename)

    def play(self, filename):
        pygame.mixer.quit()
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.stop()
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()
        pygame.mixer.quit()

        # os.remove(filename)


if __name__ == '__main__':
    v = Voice()
    v.say("Hello there, how are you doing this fine thursday evening? My name is Austin Brain, what is your name?")
    # v.clone("Hello there, how are you doing this fine thursday evening? My name is Austin Brain, what is your name?")
    # v.prompt_tts("Hello there, <laughter>how are you</laughter> doing this fine thursday evening? [breath]My name is Austin Brain, [laughter]what is your name?", "A female speaker")