import sys
import whisper
import glob
import multiprocessing as mp
import torch
import json
import os


def _compute_asr(file_path):
    output_filepath = file_path.replace(".wav", ".transcript.json")
    asr_model = getattr(_compute_asr, "asr_model", None)
    if asr_model is None:
        # Set the torch ID
        device = int(mp.current_process().name.split("-")[-1]) % torch.cuda.device_count()
        with torch.cuda.device(device):
            print(f"Loading ASR model on device {device}...")
            asr_model = whisper.load_model("large-v2")
            setattr(_compute_asr, "asr_model", asr_model)

    # result = asr_model.transcribe(file_path, beam_size=8)
    result = asr_model.transcribe(file_path)
    with open(output_filepath, "w") as fp:
        json.dump(
            {
                "text": result["text"],
                "beam_results": next(iter(result.get("segments", [{}])), {}).get("beam_results", [None])[0],
            },
            fp,
        )

    return False


if __name__ == "__main__":
    split = sys.argv[1]
    DATA_DIR = os.environ['OD3_DATA_DIRECTORY']
    ASR_MODEL = None

    # Discover all of the ASR files
    print("Discovering files for ASR...")
    asr_files = list(glob.glob(f"{DATA_DIR}/audio/{split}/**/*.wav", recursive=True))
    print(f"Discoverd {len(asr_files)} wav files")

    pool_size = torch.cuda.device_count() * 7

    with mp.Pool(pool_size) as pool:
        for sample in pool.imap_unordered(_compute_asr, asr_files):
            pass
