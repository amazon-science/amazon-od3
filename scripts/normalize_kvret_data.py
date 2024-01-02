# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import uuid
import json
import sys
import os

if __name__ == "__main__":
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]
    split = sys.argv[1]

    with open(f"{DATA_DIR}/kvret/kvret_{split}_public.json", "r") as jf:
        input_data = json.load(jf)

    outputs = []
    for input in input_data:
        driver_uuid = str(uuid.uuid4())
        assistant_uuid = str(uuid.uuid4())

        turns = []
        for tn in input["dialogue"]:
            utterance = tn["data"].pop("utterance")
            turns.append(
                {
                    "text": utterance,
                    "speaker_id": driver_uuid if tn["turn"] == "driver" else assistant_uuid,
                    "is_agent": tn["turn"] == "assistant",
                    "turn_id": str(uuid.uuid4()),
                    "meta": tn["data"],
                }
            )

        output = {
            "sample_id": str(uuid.uuid4()),
            "turns": turns,
            "source_dataset": "kvret",
            "is_llm_generated": False,
            "meta": {"scenario": input["scenario"]},
        }

        outputs.append(output)

    with open(f"{DATA_DIR}/kvret/kvret_{split}_public_converted.jsonl", "w") as jf:
        for elem in outputs:
            jf.write(f"{json.dumps(elem)}\n")
