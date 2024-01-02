# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import json
import sys
import os

if __name__ == "__main__":
    DATA_DIR = os.environ["OD3_DATA_DIRECTORY"]
    split = sys.argv[1]

    def _hash_conversation(x):
        conv = ""
        for t in x:
            conv += t["text"]

        return conv

    def _all_users_or_all_agents(x):
        return all([t["is_agent"] for t in x["turns"]]) or all([not t["is_agent"] for t in x["turns"]])

    conversations = set()
    filter_indices = []
    with open(f"{DATA_DIR}/{split}.jsonl", "r") as jf:
        for idx, line in enumerate(jf):
            data = json.loads(line)
            x = _hash_conversation(data["turns"])
            if x in conversations:
                filter_indices.append(idx)
            elif _all_users_or_all_agents(x):
                filter_indices.append(idx)
            else:
                conversations.add(x)

    filter_indices = set(filter_indices)

    # Filter the indices
    with open(f"{DATA_DIR}/{split}.jsonl", "r") as jf:
        with open(f"{DATA_DIR}/{split}_filtered.jsonl", "w") as jff:
            for idx, line in enumerate(jf):
                if idx in filter_indices:
                    continue
                jff.write(line)
