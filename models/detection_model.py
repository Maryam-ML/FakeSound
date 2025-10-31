import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from torch.autograd import Variable
import os
import sys

WORKSPACE_PATH = '/kaggle/working/FakeSound'   # Your actual path
os.environ['WORKSPACE_PATH'] = WORKSPACE_PATH
sys.path.extend([WORKSPACE_PATH])


####################################
#       Model utils
####################################
def model_init(m, method="xavier"):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")

####################################
#       MelResNet
####################################
class ResBlock(nn.Module):
    def __init__(self, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(planes)
        self.batch_norm2 = nn.BatchNorm1d(planes)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual
     
class MelResNet(nn.Module):
    def __init__(self, in_planes, planes, embed_dim=128, res_blocks=8):
        super().__init__()
        self.conv_in = nn.Conv1d(in_planes, planes, kernel_size=5, bias=False, padding=2)
        self.batch_norm = nn.BatchNorm1d(planes)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(planes))
        self.conv_out = nn.Conv1d(planes, embed_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers: x = f(x)
        x = self.conv_out(x)
        return x.transpose(1, 2)

####################################
#       BatchNorm1d
####################################
class BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.


    Example
    -------
    >>> input = torch.randn(100, 10)
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
        skip_transpose=False,
    ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(
                    shape_or[0] * shape_or[1], shape_or[3], shape_or[2]
                )

        elif not self.skip_transpose:
            x = x.transpose(-1, 1)

        x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n

####################################
#       BackTransformerEncoder
####################################
class BackTransformerEncoder(nn.Module):
    def __init__(self,
            # Transformer Parameters
            num_layers=2, nhead=4, dim_feedforward=1024, d_model=128, dropout=0.5,
            ):
        torch.nn.Module.__init__(self)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder =  nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lstm = nn.LSTM(d_model, d_model, 1, batch_first=True, bidirectional=True, dropout=0)
        
    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.lstm(x)
        return x

####################################
#       Detection Model
####################################
class BaseDetectionModel(nn.Module):
    def __init__(self, embed_dim=128, in_planes=1024, 
        # fc Parameter
        cls=1, 
        # multi task learning
        multi_task=False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.backbone = nn.Sequential(
            MelResNet(in_planes=in_planes, planes=512, embed_dim=embed_dim, res_blocks=12),
            nn.Linear(embed_dim, embed_dim),
            BatchNorm1d(input_size=embed_dim),
            BackTransformerEncoder(num_layers=2, nhead=4, dim_feedforward=1024, d_model=embed_dim, dropout=0.5)
        )
        self.back  = nn.Sequential(
            nn.ReLU(),
            # BiLSTM
            nn.Linear(embed_dim*2, cls)
        )
        if multi_task:
            self.multi_task_classifier = nn.Linear(embed_dim*2, cls)
        

    def forward(self, waveform):
        if hasattr(self, "future_extract"):
            waveform = self.future_extract(waveform)
        # waveform should be [N, T:500, D:1024/768]
        x = self.backbone(waveform)    
        pred = self.back(x)
        if hasattr(self, "multi_task_classifier"):
            pred_binary = self.multi_task_classifier(torch.mean(x, dim=1))
            return  {"pred":pred, "pred_binary":pred_binary}
        return {"pred":pred}

class WavLM_Detection(BaseDetectionModel):
    def __init__(self, embed_dim=128, in_planes=1024, multi_task=False):
        super().__init__(embed_dim=128, in_planes=in_planes, multi_task=multi_task)

        import sys
        WAVLM_PATH = f"{WORKSPACE_PATH}/models/WavLM" 
        sys.path.append(WAVLM_PATH)
        from models.WavLM.WavLM import WavLM, WavLMConfig


        # Load the pre-trained checkpoints pytorch_model.bin
        checkpoint = torch.load('/kaggle/working/FakeSound/ckpts/wavlm-large.pt')
        print(checkpoint.keys()) 
        self.cfg = WavLMConfig(checkpoint['cfg'])
 # --- Check if checkpoint has configuration ---
        if 'cfg' in checkpoint:
            print("[INFO] Using Fairseq checkpoint with 'cfg' configuration.")
            self.cfg = WavLMConfig(checkpoint['cfg'])
        elif 'config' in checkpoint:
            print("[INFO] Using Fairseq checkpoint with 'config' configuration.")
            self.cfg = WavLMConfig(checkpoint['config'])
        else:
            print("[WARNING] 'cfg' not found in checkpoint — using default configuration.")
            self.cfg = WavLMConfig()  # default config

 # --- Try loading Fairseq or fallback to Hugging Face ---
        try:
            print("[INFO] Trying to load Fairseq-style WavLM checkpoint...")
            self.wavlm = WavLM(self.cfg)
            self.wavlm.load_state_dict(checkpoint['model'], strict=False)
            print("[SUCCESS] Loaded Fairseq WavLM checkpoint successfully.")

        except (KeyError, TypeError):
            print("[WARNING] Failed to load Fairseq checkpoint — falling back to Hugging Face WavLM.")
            from transformers import WavLMModel, WavLMConfig as HFWavLMConfig
            hf_model_id = 'microsoft/wavlm-base-plus'  # you can change this if needed
            self.wavlm = WavLMModel.from_pretrained(hf_model_id)
            self.cfg = HFWavLMConfig.from_pretrained(hf_model_id)
            print(f"[SUCCESS] Loaded Hugging Face WavLM model: {hf_model_id}")

        def future_extract(self, waveform, last_layer=True):
            import torch

        # optional normalize (only if cfg has that flag)
        if getattr(self.cfg, "normalize", False):
            waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)

        # detect backend: fairseq has .extract_features; HF doesn't
        is_fairseq = hasattr(self.future_extractor, "extract_features")

        if last_layer:
           # return only the last layer features, shape [B, T, C]
            if is_fairseq:
             rep = self.future_extractor.extract_features(waveform)[0]     # fairseq
             return rep
            else:
                out = self.future_extractor(input_values=waveform)            # HF
                rep = out.last_hidden_state                                   # [B, T, C]
                return rep
        else:
            # return features from every layer
            if is_fairseq:
                rep, layer_results = self.future_extractor.extract_features(
                    waveform,
                    output_layer=getattr(self.cfg, "encoder_layers", None),
                    ret_layer_results=True
                )
                # fairseq gives tuples (x, ...) with x as [T, B, C] or [B, T, C] depending on build.
                # most forks expect [B, C, T] per layer, so transpose like your original code:
                layer_reps = [x.transpose(0, 1) for x, _ in layer_results]    # -> [B, C, T]
                return layer_reps
            else:
                # HF: ask for all hidden states
                out = self.future_extractor(input_values=waveform, output_hidden_states=True)
                # drop embedding (index 0); keep encoder layers
                hidden = list(out.hidden_states[1:])                          # each [B, T, C]
                layer_reps = [h.transpose(1, 2) for h in hidden]              # -> [B, C, T] to match fairseq path
                return layer_reps
from dataclasses import dataclass

@dataclass
class UserDirModule:
    user_dir: str

class EAT_Detection(BaseDetectionModel):
    def __init__(self, embed_dim=128, in_planes=768, multi_task=False):
        super().__init__(embed_dim=128, in_planes=in_planes, multi_task=multi_task)
        
        import sys
        FAIRSEQ_PATH = f"{WORKSPACE_PATH}/models/EAT/fairseq"
        sys.path.append(FAIRSEQ_PATH)
        import fairseq

        model_dir = f"{WORKSPACE_PATH}/models/EAT/EAT"
        checkpoint_dir = f"{WORKSPACE_PATH}/ckpts/EAT_base_epoch20.pt"
        model_path = UserDirModule(model_dir)
        fairseq.utils.import_user_module(model_path)
        self.future_extractor, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
        self.future_extractor = self.future_extractor[0]
        self.future_extractor.eval()
        self.granularity = 'frame'
   def future_extract(self, waveform):
        if self.granularity == 'frame':
            feats = self.future_extractor.extract_features(waveform, padding_mask=None, mask=False, remove_extra_tokens=True)
            feats = feats['x'].squeeze(0)
        elif self.granularity == 'utterance':
            feats = self.future_extractor.extract_features(waveform, padding_mask=None, mask=False, remove_extra_tokens=False)
            feats = feats['x']
            feats = feats[:, 0].squeeze(0)
            # Only first 500 frames need
            feats = feats[:, :500, :]
        return feats

