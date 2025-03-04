import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from models.clip_lstm_policy import CLIPLSTMPolicy

action_to_idx = {"Move forward": 0, "Turn left": 1, "Turn right": 2}

def preprocess_rgb(rgb):
    """Preprocess RGB image for CLIP: resize to 224x224 and normalize."""
    rgb = Image.fromarray(rgb)
    rgb = rgb.resize((224, 224))
    rgb = np.array(rgb) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073])  # CLIP mean
    std = np.array([0.26862954, 0.26130258, 0.27577711])  # CLIP std
    rgb = (rgb - mean) / std
    rgb = rgb.transpose(2, 0, 1)  # HWC to CHW
    return rgb.astype(np.float32)

def preprocess_depth(depth):
    """Preprocess depth map: normalize to [0,1] and resize to 224x224."""
    depth = depth / depth.max()  # Normalize to [0,1]
    depth = Image.fromarray((depth * 255).astype(np.uint8))
    depth = depth.resize((224, 224))
    depth = np.array(depth) / 255.0
    return depth[None, :, :].astype(np.float32)  # Add channel dimension

def train_on_episode(model, optimizer, instructions, rgbs, depths, action_indices, device, chunk_size=20):
    """
    Train the model on a single episode's data, processing in chunks.
    
    Args:
        model: CLIPLSTMPolicy instance
        optimizer: Torch optimizer
        instructions (list[str]): List of instruction strings
        rgbs (list[np.ndarray]): List of preprocessed RGB images
        depths (list[np.ndarray]): List of preprocessed depth maps
        action_indices (list[int]): List of action indices
        device: Torch device
        chunk_size (int): Number of timesteps per chunk
    Returns:
        avg_loss (float): Average loss for the episode
        avg_accuracy (float): Average accuracy for the episode
    """
    T = len(instructions)
    hidden = None
    total_loss = 0.0
    total_accuracy = 0.0
    num_chunks = 0

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        instr_chunk = instructions[start:end]
        rgb_chunk = rgbs[start:end]
        depth_chunk = depths[start:end]
        action_chunk = torch.tensor(action_indices[start:end], device=device)

        action_logits, new_hidden = model(instr_chunk, rgb_chunk, depth_chunk, hidden)
        loss = F.cross_entropy(action_logits, action_chunk)
        predicted_actions = torch.argmax(action_logits, dim=1)
        accuracy = (predicted_actions == action_chunk).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        hidden = (new_hidden[0].detach(), new_hidden[1].detach())
        total_loss += loss.item()
        total_accuracy += accuracy
        num_chunks += 1

        print(f"Chunk {start}-{end}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    avg_loss = total_loss / num_chunks if num_chunks > 0 else 0
    avg_accuracy = total_accuracy / num_chunks if num_chunks > 0 else 0
    print(f"Average Loss for Episode: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
    return avg_loss, avg_accuracy

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and optimizer
    model = CLIPLSTMPolicy(device).to(device)
    model_path = args.model_path
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Get list of episode directories
    training_data_dir = args.training_data_dir
    episode_dirs = [d for d in os.listdir(training_data_dir) 
                    if os.path.isdir(os.path.join(training_data_dir, d))]
    if not episode_dirs:
        print(f"No episode data found in {training_data_dir}")
        return

    # Training loop
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_losses = []
        epoch_accuracies = []

        for episode_dir in episode_dirs:
            data_dir = os.path.join(training_data_dir, episode_dir)
            print(f"Processing episode: {episode_dir}")

            # Load instructions
            with open(os.path.join(data_dir, "instructions.txt"), "r") as f:
                instructions = [line.strip() for line in f]

            # Load actions
            with open(os.path.join(data_dir, "actions.txt"), "r") as f:
                actions = [line.strip() for line in f]

            # Load RGB images as NumPy arrays
            rgb_dir = os.path.join(data_dir, "rgbs")
            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")],
                              key=lambda x: int(x.split("_")[1].split(".")[0]))
            rgbs = [np.array(Image.open(os.path.join(rgb_dir, f)).convert("RGB")) 
                    for f in rgb_files]

            # Load and preprocess depth maps
            depth_dir = os.path.join(data_dir, "depths")
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")],
                                key=lambda x: int(x.split("_")[1].split(".")[0]))
            depths = [preprocess_depth(np.load(os.path.join(depth_dir, f))) 
                      for f in depth_files]

            # Verify data consistency
            if not (len(instructions) == len(actions) == len(rgbs) == len(depths)):
                print(f"Data mismatch in {episode_dir}: instructions={len(instructions)}, "
                      f"actions={len(actions)}, rgbs={len(rgbs)}, depths={len(depths)}")
                continue

            # Convert actions to indices
            try:
                action_indices = [action_to_idx[a] for a in actions]
            except KeyError as e:
                print(f"Invalid action found in {episode_dir}: {e}")
                continue

            # Train on this episode
            avg_loss, avg_accuracy = train_on_episode(model, optimizer, instructions, 
                                                    rgbs, depths, action_indices, 
                                                    device, args.chunk_size)
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(avg_accuracy)

        # Print epoch summary
        print(f"Epoch {epoch + 1} Average Loss: {np.mean(epoch_losses):.4f}, "
              f"Average Accuracy: {np.mean(epoch_accuracies):.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP-LSTM policy from saved data.")
    parser.add_argument("--training_data_dir", type=str, default="training_data",
                        help="Directory containing training data.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs to train.")
    parser.add_argument("--chunk_size", type=int, default=40,
                        help="Chunk size for processing sequences.")
    parser.add_argument("--model_path", type=str, default="saved_model/policy.pth",
                        help="Path to save/load the model.")
    args = parser.parse_args()
    main(args)