import random 
import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
from transformers import get_scheduler
import argparse
import sys
# Set up workspace path (update this to your actual path if necessary)
WORKSPACE_PATH = "/kaggle/working"  # Example, update to your path if necessary
sys.path.extend([WORKSPACE_PATH])

#from models import detection_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deepfake audio detection model.")
    parser.add_argument(
        "--train_file", '-f', type=str, default=f"{WORKSPACE_PATH}/deepfake_data/ldm2_train.json"
    )
    parser.add_argument(
        "--batch_size", '-b', type=int, default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate", '-lr', type=float, default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_epochs", '-e', type=int, default=40,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--output_dir", '-o', type=str, default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--model_class", '-m', type=str, default="WavLM_Detection",
        help="name of model_class"
    )
    parser.add_argument(
        "--multi_task", '-mt', default=False, action='store_true',
        help="multi_task"
    )
    parser.add_argument(
        "--multi_task_ratio", '-mtr', default=0.1, type=float,
        help="multi_task"
    )
    parser.add_argument(
        "--duration", '-d', type=float, default=10,
        help="Audio duration."
    )
    parser.add_argument(
        "--time_resolution", type=float, default=0.02,
        help="."
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help="."
    )
    parser.add_argument(
        "--num_examples", '-n', type=int, default=-1,
        help="How many examples to use for training.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="A seed for reproducible training."
    )
    args = parser.parse_args()

    return args


class ASVspoofDataset(Dataset):
    def __init__(self, audio_dir, label_file, args):
        # Load label data from text file
        self.labels = pd.read_csv(label_file, header=None, sep=' ', names=['filename', 'label'])

        # Path to audio files
        self.audio_dir = audio_dir

        # Other configurations
        self.time_resolution = args.time_resolution
        self.duration = args.duration
        self.sample_rate = args.sample_rate
        num_examples = args.num_examples

        # If num_examples is not -1, limit dataset size
        if num_examples != -1:
            self.labels = self.labels[:num_examples]

    def __len__(self):
        return len(self.labels)

    def _load_wav(self, source_file):
        assert source_file.endswith('.flac'), "The file should be in FLAC format"
        wav, sr = sf.read(source_file)
        assert sr == self.sample_rate, f"Expected sample rate of {self.sample_rate}, but got {sr}"
        return torch.from_numpy(wav).float()

    def __getitem__(self, index):
        item = self.labels.iloc[index]

        # Get the file path and label
        file_path = os.path.join(self.audio_dir, item['filename'])
        audio = self._load_wav(file_path)

        # Binary label: fake -> 1, real -> 0
        binary_label = int(item['label'])

        # Create a target tensor for the onset/offset window
        tgt = np.zeros(int(self.duration / self.time_resolution))
        if binary_label == 1:
            # Assuming labels are in format 'onset_offset' like '8.28_9.96'
            [onset, offset] = item["filename"].split("_")
            tgt[int(float(onset) / self.time_resolution): int(float(offset) / self.time_resolution)] = 1

        return audio, binary_label, tgt, item["filename"]

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        batch = []
        for i in dat:
            if i == 3:
                batch.append(dat[i].tolist())
            elif i == 1:
                batch.append(np.array(dat[i]))
            else:
                batch.append(torch.tensor(np.array(dat[i].tolist()), dtype=torch.float32))
        return batch


def main():
    args = parse_args()
    print(args)

    # Set random seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Handle output directory creation
    if args.output_dir is None or args.output_dir == "":
        args.output_dir = f"{WORKSPACE_PATH}/ckpts/{args.model_class}/multi-task-{args.multi_task}_asvspoof_train"
    elif args.output_dir is not None:
        args.output_dir = f"{WORKSPACE_PATH}/ckpts/{args.model_class}/multi-task-{args.multi_task}_{args.output_dir}"
    os.makedirs(args.output_dir, exist_ok=True)
    with open("{}/summary.jsonl".format(args.output_dir), "w") as f:
        f.write(json.dumps(dict(vars(args))) + "\n\n")

    # Initialize device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = getattr(detection_model, args.model_class)(multi_task=args.multi_task).to(device)

    # Set up dataset and dataloader
    audio_dir = "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train/flac"
    label_file = "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    
    train_dataset = ASVspoofDataset(audio_dir, label_file, args)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)

    # Print the total number of audio files and labels
    num_files = len(train_dataset.labels)  # Total number of labels (rows in the CSV file)
    
    # Convert string labels ('bonafide' -> 0, 'spoof' -> 1)
    train_dataset.labels['label'] = train_dataset.labels['label'].map({
        'bonafide': 0,  # 'bonafide' corresponds to real samples
        'spoof': 1       # 'spoof' corresponds to fake samples
    })
    
    # Now sum the labels to count the fake samples (1 for fake, 0 for real)
    num_labels = sum(train_dataset.labels['label'])  # Total number of fake labels (1)
    num_real_labels = num_files - num_labels  # Total number of real labels (0)
    
    print(f"Total number of files: {num_files}")
    print(f"Total number of fake labels (1): {num_labels}")
    print(f"Total number of real labels (0): {num_real_labels}")
    

if __name__ == "__main__":
    main()




