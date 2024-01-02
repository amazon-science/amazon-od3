# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#!/bin/bash


# Normalize the KVRET data
echo "Normalizing KVRET data"
python scripts/normalize_kvret_data.py train
python scripts/normalize_kvret_data.py test
python scripts/normalize_kvret_data.py dev

# Normalize the Multi-Woz data
echo "Normalizing Multi-Woz data"
python scripts/normalize_multiwoz_data.py train
python scripts/normalize_multiwoz_data.py dev
python scripts/normalize_multiwoz_data.py test

# Normalize the NOESIS data
echo "Normalizing NOESIS data"
python scripts/normalize_noesis_data.py train
python scripts/normalize_noesis_data.py dev
python scripts/normalize_noesis_data.py test

# Normalize the SIMMC2 data
echo "Normalizing SIMMC2 data"
python scripts/normalize_simmc2_data.py dstc11_dev
python scripts/normalize_simmc2_data.py dstc11_devtest
python scripts/normalize_simmc2_data.py dstc11_train
python scripts/normalize_simmc2_data.py dstc11_teststd_public

# Normalize the DSTC-11 Track 5 data
echo "Normalizing DSTC-11 Track 5 data"
python scripts/normalize_dstc11_track5_data.py train
python scripts/normalize_dstc11_track5_data.py val
python scripts/normalize_dstc11_track5_data.py test

# Build the train/dev/test sets from the data
echo "Building datasets"
cat ./data/dstc11-track5/test_converted.jsonl ./data/kvret/kvret_test_public_converted.jsonl ./data/MultiWOZ_2.2/test_converted.jsonl  ./data/noesis/dev_converted.jsonl ./data/simmc2/dstc11_devtest_converted.jsonl > ./data/test.jsonl
cat ./data/dstc11-track5/train_converted.jsonl ./data/kvret/kvret_train_public_converted.jsonl ./data/MultiWOZ_2.2/train_converted.jsonl  ./data/noesis/train_converted.jsonl ./data/simmc2/dstc11_train_converted.jsonl > ./data/train.jsonl
cat ./data/dstc11-track5/val_converted.jsonl ./data/kvret/kvret_dev_public_converted.jsonl ./data/MultiWOZ_2.2/dev_converted.jsonl  ./data/noesis/dev_converted.jsonl ./data/simmc2/dstc11_dev_converted.jsonl > ./data/dev.jsonl

# Filter the data for identical conversations
echo "Filtering conversations"
python scripts/filter_identical_conversations.py train
python scripts/filter_identical_conversations.py dev
python scripts/filter_identical_conversations.py test

# Build the audio with TTS
echo "Building audio"
python scripts/tts_v2.py train
python scripts/tts_v2.py dev
python scripts/tts_v2.py test

# Compute the ASR
echo "Computing ASR for audio"
python scripts/asr.py train
python scripts/asr.py dev
python scripts/asr.py test

# Compute the WER for each of the splits
echo "Computing WER for each split"
python scripts/reformat_with_audio.py train
python scripts/reformat_with_audio.py dev
python scripts/reformat_with_audio.py test

# Add repeats/rephrases
echo "Adding repeats/rephrases"
python scripts/generate_repeats_rephrases.py train
python scripts/generate_repeats_rephrases.py dev
python scripts/generate_repeats_rephrases.py test
