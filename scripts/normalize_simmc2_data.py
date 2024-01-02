# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import uuid
import json
import sys
import os


if __name__ == "__main__":
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]
    split = sys.argv[1]

    with open(f"{DATA_DIR}/simmc2/simmc2.1_dials_{split}.json", "r") as jf:
        input_data = json.load(jf)

    outputs = []
    for input in input_data["dialogue_data"]:
        user_uuid = str(uuid.uuid4())
        system_uuid = str(uuid.uuid4())

        turns = []
        for tn in input["dialogue"]:
            turns.append(
                {
                    "text": tn["transcript"],
                    "speaker_id": user_uuid,
                    "is_agent": False,
                    "turn_id": str(uuid.uuid4()),
                    "meta": tn.get("transcript_annotated", {}),
                }
            )
            if "system_transcript" in tn:
                turns.append(
                    {
                        "text": tn["system_transcript"],
                        "speaker_id": system_uuid,
                        "is_agent": True,
                        "turn_id": str(uuid.uuid4()),
                        "meta": tn["system_transcript_annotated"],
                    }
                )

        output = {
            "sample_id": str(uuid.uuid4()),
            "turns": turns,
            "source_dataset": "SIMMC2.1",
            "is_llm_generated": False,
            "meta": {
                "original_dialogue_idx": input["dialogue_idx"],
                "domain": input["domain"],
                "mentioned_object_ids": input.get("mentioned_object_ids", None),
                "scene_ids": input["scene_ids"],
            },
        }

        outputs.append(output)

    with open(f"{DATA_DIR}/simmc2/{split}_converted.jsonl", "w") as jf:
        for elem in outputs:
            jf.write(f"{json.dumps(elem)}\n")
