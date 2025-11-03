import os
import json
import librosa
import numpy as np
import pandas as pd
import random
import argparse
from tqdm.auto import tqdm
from datetime import datetime
import torch

import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import SchedulerType, get_scheduler

import os
import sys

# Set the WORKSPACE_PATH to the desired directory
WORKSPACE_PATH = '/kaggle/working/FakeSound'  # Your actual path
os.environ['WORKSPACE_PATH'] = WORKSPACE_PATH  # Set the environment variable
sys.path.extend([WORKSPACE_PATH])  # Add the path to sys.path


from models import detection_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deepfake audio detection model.")
    parser.add_argument("--train_file", '-f', type=str, default=f"{WORKSPACE_PATH}/deepfake_data/ldm2_train.json")
    parser.add_argument("--batch_size", '-b', type=int, default=16)
    parser.add_argument("--learning_rate", '-lr', type=float, default=1e-4)
    parser.add_argument("--num_epochs", '-e', type=int, default=40)
    parser.add_argument("--output_dir", '-o', type=str, default=None)
    parser.add_argument("--model_class", '-m', type=str, default="WavLM_Detection")
    parser.add_argument("--multi_task", '-mt', default=False, action='store_true')
    parser.add_argument("--multi_task_ratio", '-mtr', default=0.1, type=float)
    parser.add_argument("--duration", '-d', type=float, default=10)
    parser.add_argument("--time_resolution", type=float, default=0.02)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_examples", '-n', type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"[INFO] Parsed arguments: {args}")
    return args


class DeepfakeDetectionDataset(Dataset):
    def __init__(self, data_file, args):
        print(f"[INFO] Loading dataset from {data_file}")
        self.data = json.load(open(data_file, "r"))["audios"]
        self.time_resolution = args.time_resolution
        self.duration = args.duration
        self.sample_rate = args.sample_rate
        num_examples = args.num_examples
        if num_examples != -1:
            self.data = self.data[:num_examples]
            print(f"[INFO] Using subset of dataset: {len(self.data)} samples")
        self.model_class = args.model_class
        print(f"[INFO] Dataset initialized with {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def _load_wav(self, source_file):
        #print(f"[DEBUG] Loading audio file: {source_file}")
        assert source_file.endswith('.wav'), "File must be .wav format"
        wav, sr = sf.read(source_file)
        channel = sf.info(source_file).channels
        assert channel == 1, f"Expected mono audio, got {channel} channels"
        source = torch.from_numpy(wav).float()

        if sr != 16e3: 
            #print(f"[DEBUG] Resampling from {sr} Hz to {self.sample_rate} Hz")
            source = torchaudio.functional.resample(source, orig_freq=sr, new_freq=self.sample_rate).float()  

        if self.model_class == "WavLM_Detection":
            source = torch.cat((source, torch.zeros(80)), axis=0)
        else:
            assert self.model_class == "EAT_Detection"   
            target_length, norm_mean, norm_std  = 1024, -4.268, 4.569  
            source = source - source.mean()
            source = source.unsqueeze(dim=0)
            source = torchaudio.compliance.kaldi.fbank(
                source, htk_compat=True, sample_frequency=16000, use_energy=False,
                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10
            ).unsqueeze(dim=0)
            n_frames = source.shape[1]
            diff = target_length - n_frames
            if diff > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
                source = m(source)
            elif diff < 0:
                source = source[0:target_length, :]
            source = (source - norm_mean) / (norm_std * 2)
        return source.numpy()

    def __getitem__(self, index):
        item = self.data[index]
        audio = self._load_wav(item["filepath"])
        binary_label = int(item["label"])
        tgt = np.zeros(int(self.duration / self.time_resolution))
        if binary_label == 1:
            [onset, offset] = item["onset_offset"].split("_")
            tgt[int(float(onset) / self.time_resolution): int(float(offset) / self.time_resolution)] = 1
        return audio, binary_label, tgt, item["audio_id"]

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
    print("[INFO] Starting training pipeline...")
    
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

    if args.output_dir is None or args.output_dir == "":
        args.output_dir = f"{WORKSPACE_PATH}/ckpts/{args.model_class}/multi-task-{args.multi_task}_ldm2_trainth1-4"        
    else:
        args.output_dir = f"{WORKSPACE_PATH}/ckpts/{args.model_class}/multi-task-{args.multi_task}_{args.output_dir}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {args.output_dir}")

    with open(f"{args.output_dir}/summary.jsonl", "w") as f:
        f.write(json.dumps(dict(vars(args))) + "\n\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device in use: {device}")

    print(f"[INFO] Initializing model: {args.model_class}")

    
    #model = globals()[args.model_class](multi_task=args.multi_task).to(device)

    model = getattr(detection_model, args.model_class)(multi_task=args.multi_task).to(device)
    print("[INFO] Model initialized successfully.")

    print(f"[INFO] Loading dataset from: {args.train_file}")
    train_dataset = DeepfakeDetectionDataset(args.train_file, args)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    print(f"[INFO] Number of training samples: {len(train_dataset)}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer_parameters = model.parameters()
    if hasattr(model, "future_extractor"):
        for param in model.future_extractor.parameters():
            param.requires_grad = False
        model.future_extractor.eval()
        optimizer_parameters = model.backbone.parameters()
        print("[INFO] Optimizing only backbone parameters.")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {num_trainable_parameters}")

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

            if step % 10 == 0:
                #print(f"[DEBUG] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss.item() / len(train_dataloader)
        print(f"[INFO] Epoch {epoch+1} finished — Avg loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"{args.output_dir}/best.pt")
            print(f"[INFO] New best model saved at epoch {epoch+1}")

        if epoch > 0 and epoch % 10 == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/epoch{epoch}.pt")
            print(f"[INFO] Model checkpoint saved for epoch {epoch+1}")

        result = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "best_epoch": best_epoch,
            "time": datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        }

        with open(f"{args.output_dir}/summary.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n\n")
        print(f"[INFO] Result logged for epoch {epoch+1}")

if __name__ == "__main__":
    print("[INFO] Script execution started.")
    main()
    print("[INFO] Script execution finished successfully.")
