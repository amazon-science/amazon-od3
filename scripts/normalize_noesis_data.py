# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import uuid
import json
import sys
import os

if __name__ == "__main__":
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]
    split = sys.argv[1]

    raw_path = (
        f"{DATA_DIR}/noesis/DSTC8_TESTDATA/Task_1/task-1.advising.test.blind.json"
        if split == "test"
        else f"{DATA_DIR}/noesis/DSTC8_DATA/Task_1/advising/task-1.advising.{split}.json"
    )
    with open(raw_path, "r") as jf:
        input_data = json.load(jf)

    outputs = []
    for input in input_data:
        user_uuid = str(uuid.uuid4())
        system_uuid = str(uuid.uuid4())

        turns = []
        for tn in input["messages-so-far"]:
            turns.append(
                {
                    "text": tn["utterance"],
                    "speaker_id": user_uuid if tn["speaker"] == "student" else system_uuid,
                    "is_agent": tn["speaker"] == "advisor",
                    "turn_id": str(uuid.uuid4()),
                    "meta": {},
                }
            )

        # Remove the messages
        input.pop("messages-so-far")
        input.pop("data-split")

        output = {
            "sample_id": str(uuid.uuid4()),
            "turns": turns,
            "source_dataset": "NOESIS",
            "is_llm_generated": False,
            "meta": input,
        }

        outputs.append(output)

    with open(f"{DATA_DIR}/noesis/{split}_converted.jsonl", "w") as jf:
        for elem in outputs:
            jf.write(f"{json.dumps(elem)}\n")
