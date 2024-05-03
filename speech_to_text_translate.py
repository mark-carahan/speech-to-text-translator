import argparse
import numpy as np
import sys
import time

from faster_whisper import WhisperModel, available_models
import sounddevice as sd

import ctranslate2
import sentencepiece as spm

from flores import name_to_flores_200_code, print_known_languages

# Input to the speech model.
SAMPLE_RATE = 16000
CHANNELS = 1

def run_model(model, audio_input, beam_size, sp_file=None, model_file=None, source_language=None, target_language=None, vad_parameters=None, verbose=True):
    transcribe_start_time = time.time()
    segments, info = model.transcribe(audio_input, beam_size=beam_size, vad_parameters=vad_parameters)
    transcribe_end_time = time.time()
    transcribe_duration = transcribe_end_time - transcribe_start_time
    
    if verbose:
        print("Transcribe time: {:.0f}ms".format(transcribe_duration * 1000))
        print("Detected language '%s' with probability %f" %
                (info.language, info.language_probability))

    all_text = ""
    segment_start_time = time.time()
    for segment in segments:
        if verbose:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        else:
            all_text += segment.text
    segment_end_time = time.time()
    segment_duration = segment_end_time - segment_start_time
    if verbose:
        print("Segment time: {:.0f}ms".format(segment_duration * 1000))
    else:
        print(all_text)
        
        sp = spm.SentencePieceProcessor()
        sp.load(sp_file)

        translator = ctranslate2.Translator(model_file)

        src_lang = name_to_flores_200_code(source_language)
        tgt_lang = name_to_flores_200_code(target_language)

        target_prefix = [[tgt_lang]]

        print(f"---Translating from {source_language} to {target_language}---")
        
        subworded = sp.encode_as_pieces(all_text.strip())
        subworded = [[src_lang] + subworded + ["</s>"]]
        
        translation_result = translator.translate_batch(subworded, batch_type="tokens", max_batch_size=2024, beam_size=beam_size, target_prefix=target_prefix)
        translation = translation_result[0].hypotheses[0]

        if tgt_lang in translation:
            translation.remove(tgt_lang)

        translation_text = sp.decode(translation)

        print(translation_text + "\n")
             
        
def update_audio_buffer(indata, outdata, frames, time, status):
    global audio_buffer
    if status:
        print("Sound device error: ", status, file=sys.stderr)
    indata_len = indata.shape[0]
    indata_f32 = indata.flatten()
    audio_buffer = np.concatenate((audio_buffer, indata_f32)).flatten()
    audio_buffer = audio_buffer[indata_len:]


if __name__ == "__main__":

    AVAILABLE_MODELS = available_models()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_file",
        default="../audio/two_cities.wav",
        help="File to use as input")
    parser.add_argument(
        "--microphone",
        default=None,
        help="Which microphone to use as input")
    parser.add_argument(
        "--model_size",
        default="tiny.en",
        help=f"Model to use, can be one of {', '.join(AVAILABLE_MODELS)}")
    parser.add_argument(
        "--compute_type",
        default="int8",
        help=f"Internal data type to use, can be one of int8, float16, float32, int8_float16")
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help=f"How wide an array to use for the beam search. Smaller is faster but less accurate.")
    parser.add_argument(
        "--buffer_duration",
        type=int,
        default=30,
        help=f"How long an audio buffer to retain for live transcription. Smaller is faster but less accurate.")
    parser.add_argument(
        "--model_file",
        default="../models/nllb-200-distilled-600M-int8/",
        help="Translation model")
    parser.add_argument(
        "--sp_file",
        default="../models/nllb-200-distilled-600M-int8/flores200_sacrebleu_tokenizer_spm.model",
        help="Tokenization model")
    parser.add_argument(
        "--source_language",
        default="English",
        help="Language to translate from")
    parser.add_argument(
        "--target_language",
        default="Spanish",
        help="Language to translate to")
    parser.add_argument(
        "--list_languages",
        type=bool,
        default=False,
        help="Show supported languages")

    args = parser.parse_args()

    if args.list_languages:
        print_known_languages()
        exit(0)

    AUDIO_BUFFER_DURATION_MS = int(args.buffer_duration * 1000)
    AUDIO_BUFFER_SAMPLES = int(args.buffer_duration * SAMPLE_RATE)

    model = WhisperModel(args.model_size, device="cpu",
                        compute_type=args.compute_type)
    
    if args.microphone is not None:
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = CHANNELS

    if args.microphone is None:
        run_model(model, args.audio_file, args.beam_size)
    else:
        audio_buffer = np.zeros((AUDIO_BUFFER_SAMPLES), dtype=np.float32)
        
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = CHANNELS
        with sd.Stream(channels=CHANNELS, callback=update_audio_buffer):
            while True:
                sd.sleep(1000)
                run_model(model, audio_buffer, args.beam_size, 
                          args.sp_file, 
                          args.model_file, 
                          args.source_language, 
                          args.target_language,
                          vad_parameters={
                              "threshold": 0.0,
                              "min_silence_duration_ms": AUDIO_BUFFER_DURATION_MS,
                              "min_speech_duration_ms": AUDIO_BUFFER_DURATION_MS
                              },
                          verbose=False)