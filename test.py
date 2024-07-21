from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

inputs = processor("Hello, Jean Delest. Where are you from ? ... Heum, do you want to eat something with me ?", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

# Listen the audio
# from IPython.display import Audio

# sample_rate = model.generation_config.sample_rate
# Audio(audio_array, rate=sample_rate)

# Save
sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)