# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import json
import sys
import pathlib
import os
import glob
import random
import contextlib
import time
import multiprocessing as mp

WORKER_AGENTS = 16
CUDA_DEVICES = 4


def _initialize_tts():
    tts_data = getattr(_initialize_tts, "tts_data", None)
    if tts_data is not None:
        return tts_data

    # Get the GPU device
    if "MainProcess" in mp.current_process().name:
        device = 1
        rank = 0
    else:
        device = int(mp.current_process().name.split("-")[-1]) % CUDA_DEVICES
        rank = int(mp.current_process().name.split("-")[-1]) + 1

    # A really bad fix for a race condition. We should use a mutex, but #effort.
    time.sleep(3 * rank)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    CV_DATA_DIR = os.environ["OD3_COMMONVOICE_DATA_DIRECTORY"]

    from TTS.api import TTS
    from whisper_normalizer.basic import BasicTextNormalizer
    from nemo_text_processing.text_normalization.normalize import Normalizer

    # Get a list of all of the voices
    voices = list(glob.glob(f"{CV_DATA_DIR}/en/clips/*.mp3", recursive=True))

    with contextlib.redirect_stdout(None):
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)

    uniform_agent_wav = f"{CV_DATA_DIR}/en/clips/common_voice_en_18811826.mp3"
    uniform_agent_id = os.path.splitext(os.path.basename(uniform_agent_wav))[0].split("_")[-1]

    normalizer = Normalizer(input_case="cased", lang="en")
    n2 = BasicTextNormalizer()

    tts_data = {
        "voices": voices,
        "tts": tts,
        "uniform_agent_wav": uniform_agent_wav,
        "uniform_agent_id": uniform_agent_id,
        "normalizer": normalizer,
        "n2": n2,
    }

    setattr(_initialize_tts, "tts_data", tts_data)

    return tts_data


def _tts_for_sample(sample):
    tts_data = _initialize_tts()

    sample_id = sample["sample_id"]
    sample_id_prefix = sample_id[:2]
    audio_output_dir = pathlib.Path(os.environ["OD3_DATA_DIRECTORY"]) / "audio" / split / sample_id_prefix / sample_id
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(3):
        try:
            # Remove all of the files in the output directory
            os.system(f"rm -rf {str(audio_output_dir)}/*.wav")

            agent_cloning_wav = random.choice(tts_data["voices"])
            user_cloning_wav = random.choice(tts_data["voices"])

            for turn in sample["turns"]:
                clone_path = agent_cloning_wav if turn["is_agent"] else user_cloning_wav
                turn_id = turn["turn_id"]
                normalized_text = tts_data["normalizer"].normalize(turn["text"], punct_post_process=True)
                speaker_id = os.path.splitext(os.path.basename(clone_path))[0].split("_")[
                    -1
                ]  # Gets the commonvoice ID of the speaker, and adds this to the wav file
                with contextlib.redirect_stdout(None):
                    tts_data["tts"].tts_to_file(
                        normalized_text,
                        speaker_wav=clone_path,
                        language="en",
                        file_path=str(audio_output_dir / f"{turn_id}+{speaker_id}.wav"),
                    )
                    if turn["is_agent"]:
                        tts_data["tts"].tts_to_file(
                            normalized_text,
                            speaker_wav=tts_data["uniform_agent_wav"],
                            language="en",
                            file_path=str(audio_output_dir / f'{turn_id}+{tts_data["uniform_agent_id"]}.wav'),
                        )
            break
        except Exception as ex:
            continue
    else:
        print(f"Error processing conversation {sample_id}")


if __name__ == "__main__":
    split = sys.argv[1]
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]

    samples = []
    with open(f"{DATA_DIR}/{split}_filtered.jsonl", "r") as jf:
        for s in jf:
            samples.append(json.loads(s))

    if WORKER_AGENTS > 0:
        with mp.Pool(WORKER_AGENTS) as pool:
            with open(f"{DATA_DIR}/{split}_filtered.jsonl", "r") as jf:
                for sample in pool.imap_unordered(_tts_for_sample, samples):
                    pass

    else:
        for sample in samples:
            _tts_for_sample(sample)
