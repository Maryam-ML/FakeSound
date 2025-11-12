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
from datetime import datetime
from tqdm.auto import tqdm
import librosa


# Set up workspace path (update this to your actual path if necessary)
WORKSPACE_PATH = "/kaggle/working"  # Example, update to your path if necessary
sys.path.extend([WORKSPACE_PATH])

from models import detection_model

# Define argument parser
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

    print(f"[INFO] Parsed arguments: {args}")
    return args


# ASVspoof Dataset class to load audio and labels
class ASVspoofDataset(Dataset):
    def __init__(self, audio_dir, label_file, args):
        # Load label data from text file
        # Format: speaker_id filename - - label
        # Example: LA_0079 LA_T_1138215 - - bonafide
        self.labels = pd.read_csv(
            label_file, 
            header=None, 
            sep=' ', 
            names=['speaker_id', 'filename', 'col3', 'col4', 'label']
        )
        
        # Keep only filename and label columns
        self.labels = self.labels[['filename', 'label']]

        # Path to audio files
        self.audio_dir = audio_dir

        # Other configurations
        self.time_resolution = args.time_resolution
        self.duration = args.duration
        self.sample_rate = args.sample_rate
        self.max_samples = int(self.duration * self.sample_rate)
        num_examples = args.num_examples

        # If num_examples is not -1, limit dataset size
        if num_examples != -1:
            self.labels = self.labels[:num_examples]

    def __len__(self):
        return len(self.labels)

    def _build_file_path(self, filename):
        """
        Build the correct file path for the audio file.
        The filename from the CSV is like 'LA_T_1000137' and the actual file is 'LA_T_1000137.flac'
        """
        # Check if filename already has an extension
        if not filename.endswith('.flac'):
            filename = filename + '.flac'
        
        # Build the full path
        file_path = os.path.join(self.audio_dir, filename)
        
        # Verify the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Audio file not found: {file_path}\n"
                f"Expected filename format: LA_T_XXXXXXX.flac"
            )
        
        return file_path

    def _load_audio(self, source_file):
        """
        Load audio file and pad/trim to fixed length.
        """
        # Try reading the audio file using soundfile
        try:
            wav, sr = sf.read(source_file)
        except Exception as e:
            # If soundfile fails, try librosa
            try:
                wav, sr = librosa.load(source_file, sr=self.sample_rate)
            except Exception as e2:
                print(f"[ERROR] Failed to load {source_file}")
                print(f"  soundfile error: {e}")
                print(f"  librosa error: {e2}")
                raise
        
        # Resample if necessary
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(wav).float()
        
        # Pad or trim to fixed length
        if len(audio_tensor) < self.max_samples:
            # Pad with zeros
            padding = self.max_samples - len(audio_tensor)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
        elif len(audio_tensor) > self.max_samples:
            # Trim to max_samples
            audio_tensor = audio_tensor[:self.max_samples]
        
        return audio_tensor

    def __getitem__(self, index):
        item = self.labels.iloc[index]

        # Build the correct file path with extension
        file_path = self._build_file_path(item['filename'])
        audio = self._load_audio(file_path)

        # Map string labels to binary: 'bonafide' -> 0, 'spoof' -> 1
        label_str = item['label'].strip().lower()
        if label_str == 'bonafide':
            binary_label = 0
        elif label_str == 'spoof':
            binary_label = 1
        else:
            raise ValueError(f"Unknown label: {item['label']}")

        # Create a target tensor for the onset/offset window
        # For ASVspoof dataset, we don't have onset/offset info, so just use binary label
        tgt = np.zeros(int(self.duration / self.time_resolution))
        if binary_label == 1:
            # For spoof samples, mark the entire duration
            tgt[:] = 1

        return audio, binary_label, tgt, item["filename"]

    def collate_fn(self, data):
        """
        Custom collate function to batch data properly.
        """
        audios = []
        binary_labels = []
        tgts = []
        filenames = []
        
        for audio, binary_label, tgt, filename in data:
            audios.append(audio)
            binary_labels.append(binary_label)
            tgts.append(tgt)
            filenames.append(filename)
        
        # Stack all tensors
        audios_batch = torch.stack(audios)  # Shape: (batch_size, max_samples)
        binary_labels_batch = np.array(binary_labels)  # Shape: (batch_size,)
        tgts_batch = torch.tensor(np.array(tgts), dtype=torch.float32)  # Shape: (batch_size, num_frames)
        
        return [audios_batch, binary_labels_batch, tgts_batch, filenames]


# Main function for model training
def main():
    args = parse_args()
    print("[INFO] Starting training pipeline...")

    # Set random seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("[INFO] CUDA is available, using GPU.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("[WARNING] CUDA not available, using CPU.")

    # Handle output directory creation
    if args.output_dir is None or args.output_dir == "":
        args.output_dir = f"{WORKSPACE_PATH}/ckpts/{args.model_class}/multi-task-{args.multi_task}_asvspoof_train"
    elif args.output_dir is not None:
        args.output_dir = f"{WORKSPACE_PATH}/ckpts/{args.model_class}/multi-task-{args.multi_task}_{args.output_dir}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {args.output_dir}")

    with open(f"{args.output_dir}/summary.jsonl", "w") as f:
        f.write(json.dumps(dict(vars(args))) + "\n\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device in use: {device}")

    print(f"[INFO] Initializing model: {args.model_class}")

    # model = globals()[args.model_class](multi_task=args.multi_task).to(device)
    model = getattr(detection_model, args.model_class)(multi_task=args.multi_task).to(device)
    print("[INFO] Model initialized successfully.")

    print(f"[INFO] Loading dataset from: {args.train_file}")
    audio_dir = "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train/flac"
    label_file = "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    
    train_dataset = ASVspoofDataset(audio_dir, label_file, args)
    
    # Convert string labels to numeric BEFORE creating dataloader
    # Map 'bonafide' -> 0, 'spoof' -> 1
    train_dataset.labels['label_numeric'] = train_dataset.labels['label'].map({
        'bonafide': 0,
        'spoof': 1
    })
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    print(f"[INFO] Number of training samples: {len(train_dataset)}")

    # Print the total number of audio files and labels
    num_files = len(train_dataset.labels)
    num_spoof = sum(train_dataset.labels['label_numeric'])
    num_bonafide = num_files - num_spoof
    
    print(f"Total number of files: {num_files}")
    print(f"Total number of spoof (fake) samples: {num_spoof}")
    print(f"Total number of bonafide (real) samples: {num_bonafide}")

    # Training loop setup
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer_parameters = model.parameters()
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=2e-4, eps=1e-08)

    num_update_steps_per_epoch = len(train_dataloader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=args.max_train_steps)

    print("[INFO] ***** Training Started *****")
    print(f"  Total epochs: {args.num_epochs}")
    print(f"  Total steps: {args.max_train_steps}")
    print(f"  Batch size: {args.batch_size}")

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    best_loss, best_epoch = np.inf, 0

    for epoch in range(args.num_epochs):
        print(f"\n[INFO] Starting epoch {epoch+1}/{args.num_epochs}")
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            audio, binary_label, tgt, _ = batch
            output = model(audio.to(device))
            loss = criterion(output["pred"].squeeze(-1), tgt.to(device))
            if hasattr(model, "multi_task_classifier"):
                loss = (1.0 - args.multi_task_ratio) * loss + args.multi_task_ratio * criterion(
                    output["pred_binary"].squeeze(-1), torch.tensor(binary_label, dtype=float).to(device)
                )
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.detach().float()
            progress_bar.update(1)
            completed_steps += 1
        print(f"train epoch {epoch} finish!")

        result = {}
        result["epoch"] = epoch,
        result["step"] = completed_steps
        result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)

        if result["train_loss"] < best_loss:
            best_loss = result["train_loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), f"{args.output_dir}/best.pt")
        if epoch > 0 and epoch % 10 == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/epoch{epoch}.pt")
        result["best_epoch"] = best_epoch
        print(result)
        result["time"] = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

        with open(f"{args.output_dir}/summary.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n\n")


if __name__ == "__main__":
    print("[INFO] Script execution started.")
    main()
    print("[INFO] Script execution finished successfully.")
