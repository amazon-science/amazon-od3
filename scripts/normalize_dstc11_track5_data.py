# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import uuid
import json
import sys
import os

if __name__ == "__main__":
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]
    split = sys.argv[1]

    with open(f"{DATA_DIR}/dstc11-track5/{split}/labels.json", "r") as jf:
        input_labels = json.load(jf)

    with open(f"{DATA_DIR}/dstc11-track5/{split}/logs.json", "r") as jf:
        input_logs = json.load(jf)

    outputs = []
    for labels, logs in zip(input_labels, input_logs):
        user_uuid = str(uuid.uuid4())
        system_uuid = str(uuid.uuid4())

        turns = []
        for tn in logs:
            turns.append(
                {
                    "text": tn["text"],
                    "speaker_id": user_uuid if tn["speaker"] == "U" else system_uuid,
                    "is_agent": tn["speaker"] == "S",
                    "turn_id": str(uuid.uuid4()),
                    "meta": {},
                }
            )

        # Add the response turn
        assert turns[-1]["speaker_id"] == user_uuid

        if labels["target"]:
            try:
                turns.append(
                    {
                        "text": labels["response"],
                        "speaker_id": system_uuid,
                        "is_agent": True,
                        "turn_id": str(uuid.uuid4()),
                        "meta": {},
                    }
                )
            except KeyError as ex:
                print(labels, logs)
                raise ex

        output = {
            "sample_id": str(uuid.uuid4()),
            "turns": turns,
            "source_dataset": "DSCT11-Track-5",
            "is_llm_generated": False,
            "meta": {
                "knowledge": labels.get("knowledge", None),
                "target": labels["target"],
            },
        }

        outputs.append(output)

    with open(f"{DATA_DIR}/dstc11-track5/{split}_converted.jsonl", "w") as jf:
        for elem in outputs:
            jf.write(f"{json.dumps(elem)}\n")
