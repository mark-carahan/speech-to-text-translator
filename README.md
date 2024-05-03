# Speech to Text Translator using Raspberry Pi 5
This is a quick project that I created by incorportating two labs from [Pete Warden's EE292D](https://github.com/ee292d/labs) class. It incorporates lab4 and lab5 from the repository. Make sure you go to his [lab0](https://github.com/ee292d/labs/tree/main/lab0) to get the Raspberry Pi 5 setup correctly for this implementation.

It will show you how to recognize speech from an audio file or microphone using OpenAI's Whisper model. It also then translates the speech into [over 200 languages](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) using NLLB (No Language Left Behind) model from Meta. The downloaded model will be 500MB big, so make sure you have enough space on your Raspberry Pi SD Card before going to the second step.

Here are the steps to make it work (it assumes that you downloaded this repository in your home directory) and example usage:

1) Lets clone this git repository:

```bash
cd ~
git clone https://github.com/mark-carahan/speech-to-text-translator.git
```

2) Download the model:

```bash
cd ~/speech-to-text-translator
./download_model.sh
```

3) Lets create a virtual environment so we can encapsulate the runtimes:

```bash
sudo -H python -m pip install --break-system-packages virtualenv
python -m virtualenv env
source env/bin/activate
```

4) Once the model is downloaded, you'll need to install the Python dependencies and system packages for running it:

```bash
sudo apt install -y libportaudio2
pip install --break-system-packages ctranslate2 transformers sentencepiece faster-whisper sounddevice
```

5) Example usage

This is if you want to input a .wav file into the Python file and have it convert to text (speech to text). It uses the example .wav file in the audio folder:

```bash
python speech_to_text_translate.py --audio_file=../audio/two_cities.wav
```

This is if you want to use the full speech to text and translate functionality. It assumes that you are using the default microphone (usually correct if you have just one installed), a `beam_size` of 1, and `buffer_duration` of 5:

```bash
python speech_to_text_translate.py --microphone=default --beam_size=1 --buffer_duration=5 --source_language="English" --target_language="Spanish"
```
You can make changes to the `beam_size` and `buffer_duration` and see how slow or fast the translation works. Make sure that the `source_language` and `target_language` are both part of [this list](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200).