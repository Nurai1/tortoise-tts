import os
os.environ["CURL_CA_BUNDLE"]=""

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

tts = TextToSpeech()


# Define your own voice folder
VOICE_NAME='rick'
text='Hello from this tutorial, I hope you enjoy it'

# Generate with your own voice
voice_samples, conditioning_latents = load_voice(VOICE_NAME)
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                          preset=preset)
torchaudio.save(f'generated-{VOICE_NAME}.wav', gen.squeeze(0).cpu(), 24000)
IPython.display.Audio(f'generated-{VOICE_NAME}.wav')
