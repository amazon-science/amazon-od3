# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import sys
import sys
import json
import glob
from nemo_text_processing.text_normalization.normalize import Normalizer
from whisper_normalizer.basic import BasicTextNormalizer
from jiwer import wer
import os
import multiprocessing as mp

UNIVERSAL_AGENT_VOICE_ID = "18811826"
WORKERS = 32


def _load_audios(sample_id, turn_id, turn_text, n1, n2):
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]

    # Find the wav files
    wav_files = list(glob.glob(f"{DATA_DIR}/audio/{split}/{sample_id[:2]}/{sample_id}/{turn_id}*.wav"))

    # Extract all of the
    outputs = {}
    for w in wav_files:
        # Find the matching transcript file
        turn_fragment = os.path.splitext(os.path.basename(w))[0]
        voice_id = turn_fragment.split("+")[-1]

        if os.path.exists(f"{DATA_DIR}/audio/{split}/{sample_id[:2]}/{sample_id}/{turn_fragment}.transcript.json"):
            transcript_file = f"{DATA_DIR}/audio/{split}/{sample_id[:2]}/{sample_id}/{turn_fragment}.transcript.json"
            with open(transcript_file, "r") as tf:
                dt = json.load(tf)
            asr_transcript = dt["text"]
            nbest = dt["beam_results"]
        elif os.path.exists(f"{DATA_DIR}/audio/{split}/{sample_id[:2]}/{sample_id}/{turn_fragment}.transcript"):
            transcript_file = f"{DATA_DIR}/audio/{split}/{sample_id[:2]}/{sample_id}/{turn_fragment}.transcript"
            with open(transcript_file, "r") as tf:
                asr_transcript = tf.read().strip()
                nbest = None
        else:
            asr_transcript = None
            nbest = None

        if asr_transcript is not None:
            gt_norm = n2(n1.normalize(turn_text, punct_post_process=True))
            pd_norm = n2(n1.normalize(asr_transcript, punct_post_process=True))

            # Compute the WER
            try:
                error_rate = wer(gt_norm, pd_norm)
            except ValueError as ex:
                print("ERROR", gt_norm, pd_norm)
                error_rate = None
        else:
            gt_norm = n2(n1.normalize(turn_text, punct_post_process=True))
            pd_norm = None
            error_rate = None

        outputs[voice_id] = {
            "path": f"{split}/{sample_id[:2]}/{sample_id}/{turn_fragment}.wav",
            "voice_id": voice_id,
            "universal_agent": UNIVERSAL_AGENT_VOICE_ID == voice_id,
            "asr_transcript": asr_transcript,
            "normalized_ground_truth": gt_norm,
            "normalized_asr_transcript": pd_norm,
            "word_error_rate": error_rate,
            "nbest": nbest,
        }

    return outputs


def _compute_updated_sample(s):
    try:
        normalizers = getattr(_compute_updated_sample, "normalizers", None)
        if normalizers is None:
            n1 = Normalizer(input_case="cased", lang="en")
            n2 = BasicTextNormalizer()
            setattr(_compute_updated_sample, "normalizers", (n1, n2))
        else:
            n1, n2 = normalizers

        output_turns = []
        for turn in s["turns"]:
            turn_audios = _load_audios(s["sample_id"], turn["turn_id"], turn["text"], n1, n2)
            output_turns.append({**turn, "audio": turn_audios})

        s["turns"] = output_turns
    except Exception as ex:
        print("Error", ex)
        return None

    return s


# Load the jsonl files
if __name__ == "__main__":
    split = sys.argv[1]
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]

    samples = []
    with open(f"{DATA_DIR}/{split}_filtered.jsonl", "r") as jf:
        for line in jf:
            samples.append(json.loads(line))

    output_samples = []
    with mp.Pool(WORKERS) as pool:
        for p in pool.imap_unordered(_compute_updated_sample, samples):
            if p is not None:
                output_samples.append(p)

    # NOTE: TEMPORARY -- REMOVE LATER
    with open(f"{DATA_DIR}/{split}_filtered_audio_augmented.jsonl", "w") as txtf:
        for s in output_samples:
            txtf.write(f"{json.dumps(s)}\n")
