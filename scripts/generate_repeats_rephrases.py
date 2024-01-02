# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import json
import sys
import random
import multiprocessing as mp
import logging
import uuid
import pathlib
import os
import re
import contextlib
import time


MODEL_STRING = "mosaicml/mpt-30b"
CUDA_DEVICES = 4

if sys.argv[1].strip() == "dev_mini":
    REPHRASE_INTRODUCE_RATE = 1
    REPHRASE_INTRODUCE_WER_LIMIT = 0
else:
    REPHRASE_INTRODUCE_RATE = 0.25
    REPHRASE_INTRODUCE_WER_LIMIT = 0.15

# Setup logging with multiprocessing process ID
log_format = "[%(asctime)s] [Process ID: %(process)d] [%(levelname)s] [%(name)s] - %(message)s"
logging.basicConfig(format=log_format, datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO)


def _initialize_tts():
    # Get the GPU device
    if "MainProcess" in mp.current_process().name:
        device = 1
    else:
        device = int(mp.current_process().name.split("-")[-1]) % CUDA_DEVICES

    # A really bad fix for a race condition. We should use a mutex, but #effort.
    time.sleep(5 * device)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    CV_DATA_DIR = os.environ["OD3_COMMONVOICE_DATA_DIRECTORY"]

    from TTS.api import TTS
    from whisper_normalizer.basic import BasicTextNormalizer
    from accelerate import infer_auto_device_map
    from nemo_text_processing.text_normalization.normalize import Normalizer

    tts_data = getattr(_initialize_tts, "tts_data", None)
    if tts_data is not None:
        return tts_data

    with contextlib.redirect_stdout(None):
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)

    uniform_agent_wav = f"{CV_DATA_DIR}/en/clips/common_voice_en_18811826.mp3"
    uniform_agent_id = os.path.splitext(os.path.basename(uniform_agent_wav))[0].split("_")[-1]

    normalizer = Normalizer(input_case="cased", lang="en")
    n2 = BasicTextNormalizer()

    tts_data = {
        # 'voices': voices,
        "tts": tts,
        "uniform_agent_wav": uniform_agent_wav,
        "uniform_agent_id": uniform_agent_id,
        "normalizer": normalizer,
        "n2": n2,
    }

    setattr(_initialize_tts, "tts_data", tts_data)

    return tts_data


def _init_lm():
    global MODEL_STRING

    # Get the GPU device
    if "MainProcess" in mp.current_process().name:
        device = 1
    else:
        device = int(mp.current_process().name.split("-")[-1]) % CUDA_DEVICES

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    import torch

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_STRING,
        device_map="auto",
        quantization_config=nf4_config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_STRING,
        trust_remote_code=True,
        padding_side="left",
    )

    tokenizer.pad_token_id = model.config.eos_token_id or tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        logging.debug("Warning: Pad Token is None, Setting batch size to 1")
        # batch_size = 1

    split_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        max_new_tokens=32,
    )

    return {
        "pipe": pipe,
        "split_token": split_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def _rephrase(input_string):
    language_model = getattr(_rephrase, "language_model", None)
    if language_model is None:
        language_model = _init_lm()
        setattr(_rephrase, "language_model", language_model)

    prompt = f'Our automated speech recognition model found "{input_string}" hard to parse, so we rephrased it to use easier to understand words as "'

    for _ in range(5):
        output = language_model["pipe"](
            prompt,
            pad_token_id=language_model["pad_token_id"],
            eos_token_id=language_model["eos_token_id"],
            return_full_text=False,
            do_sample=True,
            temperature=0.8,
            top_k=5,
            top_p=0.95,
        )

        # Clean up the generated text
        generated_text = output[0]["generated_text"].strip().replace("\n", " ").replace("\t", " ").split('"')[0]
        if generated_text.strip().lower() != input_string.strip().lower():
            break

    logging.debug('Rephrased "%s" as "%s"', input_string, generated_text)
    return generated_text


def _get_agent_unknown_variant(input_string):
    language_model = getattr(_rephrase, "language_model", None)
    if language_model is None:
        language_model = _init_lm()
        setattr(_rephrase, "language_model", language_model)

    prompt = f'Our ASR model found the audio unintelligible, so the agent said in response: "'

    output = language_model["pipe"](
        prompt,
        pad_token_id=language_model["pad_token_id"],
        eos_token_id=language_model["eos_token_id"],
        return_full_text=False,
        do_sample=True,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
    )

    # Clean up the generated text
    generated_text = output[0]["generated_text"].strip().replace("\n", " ").replace("\t", " ").split('"')[0]
    # Get the first sentence in the output
    if match := re.match(r".*?[\.!?]", generated_text):
        generated_text = match.group(0)

    logging.debug('Generated unknown agent variant for "%s": "%s"', input_string, generated_text)
    return generated_text


def _generate_repeat(sample, turn_idx, agent_response):
    # Generate a new turn ID
    tts = _initialize_tts()
    query_turn = sample["turns"][turn_idx]
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]
    CV_DATA_DIR = os.environ["OD3_COMMONVOICE_DATA_DIRECTORY"]

    # Generate the TTS of the turn IDS
    query_audio = list(query_turn["audio"].values())[0]
    voice_id = query_audio["voice_id"]
    voice = f"{CV_DATA_DIR}/en/clips/common_voice_en_{voice_id}.mp3"

    sample_id = sample["sample_id"]
    sample_id_prefix = sample_id[:2]
    audio_output_dir = pathlib.Path(DATA_DIR) / "audio" / split / sample_id_prefix / sample_id
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    turn_id = str(uuid.uuid4())
    normalized_text = tts["normalizer"].normalize(query_turn["text"], punct_post_process=True)
    with contextlib.redirect_stdout(None):
        tts["tts"].tts_to_file(
            normalized_text,
            speaker_wav=voice,
            language="en",
            file_path=str(audio_output_dir / f"{turn_id}+{voice_id}.wav"),
        )

    # We assume that ASR is correct for all repeats/rephrases... We don't have to do this, but this is something we can
    # do to avoid introducing all kinds of weird learning artifacts.
    output_turn = {
        "text": query_turn["text"],
        "speaker_id": query_turn["speaker_id"],
        "is_agent": False,
        "turn_id": turn_id,
        "meta": {
            **query_turn["meta"],
            "is_induced_by_repeat_rephrase": True,
            "is_machine_generated": True,
            "repeat_rephrase_type": "repeat",
        },
        "audio": {
            voice_id: {
                "path": f"{split}/{sample_id_prefix}/{sample_id}/{turn_id}+{voice_id}.wav",
                "voice_id": voice_id,
                "universal_agent": False,
                "asr_transcript": query_audio["asr_transcript"],
                "normalized_ground_truth": query_audio["normalized_ground_truth"],
                "normalized_asr_transcript": query_audio["normalized_ground_truth"],
                "word_error_rate": 0.0,
                "nbest": query_audio["nbest"],
            }
        },
        "turn_is_repeat_rephrase": True,
    }

    return output_turn


def _generate_rephrase(sample, turn_idx, agent_response):
    # Generate a new turn ID
    tts = _initialize_tts()
    query_turn = sample["turns"][turn_idx]
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]
    CV_DATA_DIR = os.environ["OD3_COMMONVOICE_DATA_DIRECTORY"]

    # Generate the TTS of the turn IDS
    query_audio = list(query_turn["audio"].values())[0]
    voice_id = query_audio["voice_id"]
    voice = f"{CV_DATA_DIR}/en/clips/common_voice_en_{voice_id}.mp3"

    sample_id = sample["sample_id"]
    sample_id_prefix = sample_id[:2]
    audio_output_dir = pathlib.Path(DATA_DIR) / "audio" / split / sample_id_prefix / sample_id
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    turn_id = str(uuid.uuid4())

    # Determine what the rephrase should be
    rephrase_text = _rephrase(query_turn["text"])
    normalized_output = tts["normalizer"].normalize(rephrase_text, punct_post_process=True)
    normalized_ground_truth = tts["n2"](normalized_output)
    with contextlib.redirect_stdout(None):
        tts["tts"].tts_to_file(
            normalized_output,
            speaker_wav=voice,
            language="en",
            file_path=str(audio_output_dir / f"{turn_id}+{voice_id}.wav"),
        )

    # We assume that ASR is correct for all repeats/rephrases... We don't have to do this, but this is something we can
    # do to avoid introducing all kinds of weird learning artifacts.
    output_turn = {
        "text": rephrase_text,
        "speaker_id": query_turn["speaker_id"],
        "is_agent": False,
        "turn_id": turn_id,
        "meta": {
            **query_turn["meta"],
            "is_induced_by_repeat_rephrase": True,
            "is_machine_generated": True,
            "repeat_rephrase_type": "rephrase",
        },
        "audio": {
            voice_id: {
                "path": f"{split}/{sample_id_prefix}/{sample_id}/{turn_id}+{voice_id}.wav",
                "voice_id": voice_id,
                "universal_agent": False,
                "asr_transcript": rephrase_text,
                "normalized_ground_truth": normalized_ground_truth,
                "normalized_asr_transcript": normalized_ground_truth,
                "word_error_rate": 0.0,
                "nbest": query_audio["nbest"],
            }
        },
        "turn_is_repeat_rephrase": True,
    }

    return output_turn


def _generate_agent_response_turn(sample, turn_idx):
    # Generate a new turn ID
    tts = _initialize_tts()
    query_turn = sample["turns"][turn_idx]
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]
    CV_DATA_DIR = os.environ["OD3_COMMONVOICE_DATA_DIRECTORY"]

    # Get the first agent turn in the sample, and get the agent voices
    first_agent_turn = [t for t in sample["turns"] if t["is_agent"]][0]

    # Generate new response turn information
    turn_id = str(uuid.uuid4())

    # Response text
    # response_text = "I'm sorry, I don't understand."
    response_text = _get_agent_unknown_variant(query_turn["text"])

    # For each voice
    audios = {}
    for voice_id, values in first_agent_turn["audio"].items():
        # Generate the TTS of the turn IDS
        voice = f"{CV_DATA_DIR}/en/clips/common_voice_en_{voice_id}.mp3"

        sample_id = sample["sample_id"]
        sample_id_prefix = sample_id[:2]
        audio_output_dir = pathlib.Path(DATA_DIR) / "audio" / split / sample_id_prefix / sample_id
        audio_output_dir.mkdir(parents=True, exist_ok=True)

        normalized_output = tts["normalizer"].normalize(response_text, punct_post_process=True)
        normalized_ground_truth = tts["n2"](normalized_output)
        with contextlib.redirect_stdout(None):
            tts["tts"].tts_to_file(
                normalized_output,
                speaker_wav=voice,
                language="en",
                file_path=str(audio_output_dir / f"{turn_id}+{voice_id}.wav"),
            )

        audios[voice_id] = {
            "path": f"{split}/{sample_id_prefix}/{sample_id}/{turn_id}+{voice_id}.wav",
            "voice_id": voice_id,
            "universal_agent": values["universal_agent"],
            "asr_transcript": response_text,
            "normalized_ground_truth": normalized_ground_truth,
            "normalized_asr_transcript": normalized_ground_truth,
            "word_error_rate": 0.0,
            "nbest": [response_text for _ in range(8)],
        }

    # We assume that ASR is correct for all repeats/rephrases... We don't have to do this, but this is something we can
    # do to avoid introducing all kinds of weird learning artifacts.
    output_turn = {
        "text": response_text,
        "speaker_id": first_agent_turn["speaker_id"],
        "is_agent": True,
        "turn_id": turn_id,
        "meta": {
            **query_turn["meta"],
            "is_induced_by_repeat_rephrase": True,
            "is_machine_generated": True,
            "repeat_rephrase_type": None,
        },
        "audio": audios,
        "turn_is_repeat_rephrase": False,
    }

    return output_turn


def _generate_repeat_rephrase(sample, turn_idx):
    # Need to generate two things, the agent response, and the repeat/rephrase.

    agent_response = _generate_agent_response_turn(sample, turn_idx)
    if random.random() < 0.5:
        # Generate a repeat
        user_response = _generate_repeat(sample, turn_idx, agent_response)
    else:
        user_response = _generate_rephrase(sample, turn_idx, agent_response)

    return (agent_response, user_response)


def _augment_with_repeat_or_rephrase(sample):
    try:
        new_turns = []
        session_has_repeat_rephrase = False
        for idx, turn in enumerate(sample["turns"]):
            turn["meta"] = {
                **turn["meta"],
                "is_induced_by_repeat_rephrase": turn["meta"].get("is_induced_by_repeat_rephrase", False),
                "is_machine_generated": turn["meta"].get("is_machine_generated", False),
                "repeat_rephrase_type": turn["meta"].get("repeat_rephrase_type", None),
            }

            # Add the turn to the audio
            new_turns.append(
                {
                    **turn,
                    "turn_is_repeat_rephrase": False,
                }
            )

            if turn["is_agent"] or session_has_repeat_rephrase:
                continue

            # Look at the turn audio ASR, if the WER is bad, then we need to introduce a repeat/rephrase
            try:
                turn_audio = list(turn["audio"].values())[0]
            except (IndexError, KeyError) as ex:
                # There's no audio for this sample, so we can't introduce a rp/rf
                continue

            if (
                turn_audio["word_error_rate"] is not None
                and turn_audio["word_error_rate"] > REPHRASE_INTRODUCE_WER_LIMIT
            ):
                if random.random() < REPHRASE_INTRODUCE_RATE:
                    new_turns.extend(_generate_repeat_rephrase(sample, idx))
                    session_has_repeat_rephrase = True

        sample["turns"] = new_turns

        return sample

    except IndexError as ex:
        # This is probably caused by a conversation consisting of ONLY user or ONLY agent turns.
        return None


if __name__ == "__main__":
    split = sys.argv[1]
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]

    # Load the JSONL conversations
    samples = []
    with open(f"{DATA_DIR}/{split}_filtered_audio_augmented.jsonl", "r") as txtf:
        for line in txtf:
            samples.append(json.loads(line))

    output_samples = []
    with mp.Pool(CUDA_DEVICES) as pool:
        finished_samples = 0
        for s in pool.imap_unordered(_augment_with_repeat_or_rephrase, samples):
            if s is not None:
                output_samples.append(s)
            finished_samples += 1
            if finished_samples % 100 == 0:
                print(f"Finished {finished_samples}/{len(samples)}")

    with open(f"{DATA_DIR}/{split}_repeat_rephrase.jsonl", "w") as txtf:
        for s in output_samples:
            txtf.write(f"{json.dumps(s)}\n")
