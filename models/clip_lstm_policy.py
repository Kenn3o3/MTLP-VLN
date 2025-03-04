import torch
import torch.nn as nn
import clip
from PIL import Image
import numpy as np
import torch.nn.functional as F

class DepthCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 16 * 16, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CLIPLSTMPolicy(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.depth_cnn = DepthCNN().to(device)
        self.lstm = nn.LSTM(input_size=512 + 512 + 128, hidden_size=256, num_layers=1)
        self.action_head = nn.Linear(256, 3)  # Updated to 3 actions
        self.to(device)

    def forward(self, instructions, rgbs, depths, hidden=None):
        """
        Process a sequence of instructions and observations.
        Args:
            instructions (list[str]): List of instruction strings, length T
            rgbs (list[np.ndarray]): List of RGB images, length T
            depths (list[np.ndarray]): List of depth images, length T
            hidden (tuple): LSTM hidden state (h_0, c_0), optional
        Returns:
            torch.Tensor: Action logits, shape [T, 3]
            tuple: Updated hidden state (h_n, c_n)
        """
        T = len(instructions)

        # Batch encode all instructions
        all_tokens = clip.tokenize(instructions).to(self.device)  # [T, 77]
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(all_tokens)  # [T, 512]

        # Batch process RGB features
        batch_size = 10
        rgb_feats = []
        for i in range(0, T, batch_size):
            batch_rgbs = rgbs[i:i + batch_size]
            rgb_tensors = [self.preprocess(Image.fromarray(rgb)).to(self.device) for rgb in batch_rgbs]
            rgb_tensors = torch.stack(rgb_tensors)
            with torch.no_grad():
                batch_feats = self.clip_model.encode_image(rgb_tensors)
            rgb_feats.append(batch_feats)
        rgb_feats = torch.cat(rgb_feats, dim=0)  # [T, 512]

        # Batch process depth features
        depth_feats = []
        for i in range(0, T, batch_size):
            batch_depths = depths[i:i + batch_size]
            depth_tensors = torch.stack([torch.from_numpy(depth).float() for depth in batch_depths], dim=0).to(self.device)  # [N, 1, 224, 224]
            depth_tensors = F.interpolate(depth_tensors, size=(256, 256), mode='nearest')  # [N, 1, 256, 256]
            batch_feats = self.depth_cnn(depth_tensors)
            depth_feats.append(batch_feats)
        depth_feats = torch.cat(depth_feats, dim=0)  # [T, 128]

        # Concatenate features
        feats = torch.cat([text_feats, rgb_feats, depth_feats], dim=1)  # [T, 1152]

        # Process with LSTM
        lstm_out, new_hidden = self.lstm(feats, hidden)  # [T, 256]
        action_logits = self.action_head(lstm_out)  # [T, 3]
        return action_logits, new_hidden