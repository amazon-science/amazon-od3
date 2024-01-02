# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import uuid
import json
import glob
import sys
import os

if __name__ == "__main__":
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]
    split = sys.argv[1]

    input_data = []
    for input_file in glob.glob(f"{DATA_DIR}/MultiWOZ_2.2/{split}/*.json"):
        with open(input_file, "r") as jf:
            input_data.extend(json.load(jf))

    outputs = []
    for input in input_data:
        user_uuid = str(uuid.uuid4())
        system_uuid = str(uuid.uuid4())

        turns = []
        for tn in input["turns"]:
            turns.append(
                {
                    "text": tn["utterance"],
                    "speaker_id": user_uuid if tn["speaker"] == "USER" else system_uuid,
                    "is_agent": tn["speaker"] == "SYSTEM",
                    "turn_id": str(uuid.uuid4()),
                    "meta": {"frames": tn["frames"]},
                }
            )

        output = {
            "sample_id": str(uuid.uuid4()),
            "turns": turns,
            "source_dataset": "MultiWOZ_2.2",
            "is_llm_generated": False,
            "meta": {"original_dialogue_id": input["dialogue_id"], "services": input["services"]},
        }

        outputs.append(output)

    with open(f"{DATA_DIR}/MultiWOZ_2.2/{split}_converted.jsonl", "w") as jf:
        for elem in outputs:
            jf.write(f"{json.dumps(elem)}\n")
