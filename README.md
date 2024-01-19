<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->

# OD3: Open Directed Dialogue Dataset

OD3 (Open Directed Dialogue Dataset), is designed to allow the community to explore further research into leveraging flawed conversational interactions to improve model performance. OD3 is a collection of 63K conversations (600K turns, 1,172 hours of audio) drawn from existing natural language task-oriented dialog datasets, and augmented with synthetic audio. OD3 is further augmented with turns containing repeats and rephrases of previous failed utterances. This repository contains the data and scripts for the OD3 dataset.

For more details, check out [the paper](https://arxiv.org/abs/2401.02417)!

# Latest News

-   [January 16, 2024] Data available for download!
-   [January 5, 2024] [OD3 Paper](https://arxiv.org/abs/2401.02417) available on ArXiV!

# Downloading the Dataset and Code

The dataset used in the paper can be downloaded following the instructions below. Alternatively, the dataset can be rebuilt using the code released in this repository (if more synthetic data needs to be added, or
if you'd like to use a different language model than released in the original dataset).

## Downloading the dataset (v1.0)

To download/prepare the dataset, first clone this repository, and change into the data directory:
```bash
git clone https://github.com/amazon-science/amazon-od3.git
cd amazon-od3/data/
```
Next, download the annotation files:
```bash
# Training Annotations (1017MB)
curl -O https://da4reusta8pbu.cloudfront.net/od3_train.jsonl
# Validation Annotations (148MB)
curl -O https://da4reusta8pbu.cloudfront.net/od3_dev.jsonl
# Test Annotations (192MB)
curl -O https://da4reusta8pbu.cloudfront.net/od3_test.jsonl
# Audio Files (94GB)
curl -O https://da4reusta8pbu.cloudfront.net/audio.tar.gz
```
Finally, extract the audio files:
```bash
tar -xzvf ./audio.tar.gz
```

## Building a new version of the synthetic data

To build a new version of the synthetic data, first clone the repository.

```
git clone https://github.com/amazon-science/amazon-od3
```

Next, you need to obtain the seed datasets from their local repositories, and place them in the `data/` folder:

-   [DSTC11 (Track 5)](https://github.com/alexa/dstc11-track5)
-   [KVRET](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)
-   [MultiWoz (2.2)](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2)
-   [NOESIS](https://github.com/dstc8-track2/NOESIS-II)
-   [SIMMC (2)](https://github.com/facebookresearch/simmc2/tree/main/data)

You also need to obtain the CommonVoice Corpus [cv-corpus-14.0-2023-06-23](https://commonvoice.mozilla.org/en/datasets) which is used for voice cloning. Make sure to update the path to the corpus in the next step.

At the end of this step, your data folder should look something like the following:

```
data/
    dstc11-track5/
        test/
        train/
        val/
        DATALICENSE
        knowledge.json
        output_schema.json
        README.md
    kvret/
        kvret_dev_public.json
        kvret_entities.json
        kvret_test_public.json
        kvret_train_public.json
    MultiWOZ_2.2/
        dev/
        test/
        train/
        schema.json
        dialog_acts.json
        README.md
    noesis/
        DSTC8_DATA/
            External_data/
            Supplementary_data/
            Task_1/
            Task_2/
            Task_3/
            Task_4/
        DSTC8_TESTDATA/
            Task_1/
            Task_2/
            Task_3/
            Task_4/
        DSTC8_TESTDATA_GROUNDTRUTH/
            ubuntu-subtask-4/
            advising-task-1.txt
            advising-task-3.txt
            ubuntu-task-1.txt
            ubuntu-task-2.txt
    simmc2/
        fashion_prefab_metadata_all.json
        furniture_prefab_metadata_all.json
        LICENSE
        README.md
        simmc2_dials_dstc10_dev_retrieval_candidates.json
        simmc2_dials_dstc10_dev.json
        simmc2_dials_dstc10_devtest_retrieval_candidates.json
        simmc2_dials_dstc10_devtest.json
        simmc2_dials_dstc10_teststd_public.json
        simmc2_dials_dstc10_teststd_retrieval_candidates_public.json
        simmc2_dials_dstc10_train.json
        simmc2.1_dials_dstc11_dev.json
        simmc2.1_dials_dstc11_devtest.json
        simmc2.1_dials_dstc11_mini.json
        simmc2.1_dials_dstc11_teststd_public.json
        simmc2.1_dials_dstc11_train.json
```

Then install the requirements, and run the `build_dataset.sh` script.

```
pip install -r requirements.txt
chmod +x ./build_dataset.sh
export OD3_DATA_DIRECTORY="<Path To Data Directory, no trailing slash>"
export OD3_COMMONVOICE_DATA_DIRECTORY="<Path to CommonVoice corpus, no trailing slash>"
./build_dataset.sh
```

# Contact

For questions related to the OD3 dataset, contact [davidchan@berkeley.edu](mailto:davidchan@berkeley.edu).

# Citations

If you want to publish experimental results with our datasets, please cite the following articles:

```
@inproceedings{chan2023domain,
  title={Task Oriented Dialogue as a Catalysis for Self-Supervised Automatic Speech Recognition},
  author={Chan, David M and Ghosh, Shalini and Tulsiani, Hitesh and Rastrow, Ariya and Hoffmeister, Bj{\"o}rn},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```

# License

OD3 is released under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode), see [LICENSE](LICENSE) for details.
