from pathlib import Path
import sys
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

def get_gif(name_gif):
    gif_path= ROOT / 'gif'
    return str(Path('.') / gif_path / name_gif)

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 50,
        "lr": 10**-4,
        "seq_len": 400,
        "d_model": 512,
        "question": "question",
        "answer": "answer",
        "model_folder": ROOT/"weights",
        "data_path": ROOT/"transformer/data_train.jsonl",
        "save_conversation": 'current_data.json',
        "model_basename": "tmodel_",
        "preload": '05',
        "tokenizer_file":  str(ROOT/"transformer/tokenizer_{0}.json"),
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
