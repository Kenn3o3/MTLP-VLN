import torch
import base64
import cv2
import numpy as np
from openai import OpenAI
from torch import nn
from typing import List, Tuple

class VLNModel(nn.Module):
    """Vision-Language Navigation model."""
    def __init__(self, device: str = "cuda", api_key: str = None):
        super().__init__()
        self.device = device
        self.llm = LLMReasoner(api_key)
        self.subtasks: List[str] = []
        self.current_subtask_idx: int = 0
        self.reasoning_history: List[str] = []
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.actions = ["Move forward", "Turn right", "Turn left", "Stop moving"]
        self.dummy_param = nn.Parameter(torch.zeros(1))  # For nn.Module compliance
        self.all_subtasks_completed = False

    def _colorize_depth(self, depth_np: np.ndarray) -> np.ndarray:
        """Convert depth map to a colorized image using JET colormap."""
        depth_norm = np.clip(depth_np, 0.5, 5.0)  # Depth range: 0.5m to 5m
        depth_norm = (depth_norm - 0.5) / 4.5  # Normalize to [0,1]
        depth_colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return depth_colormap

    def decompose_instruction(self, instruction_text: str) -> None:
        """Decompose instruction and reset subtask tracking."""
        self.subtasks = self.llm.decompose_instruction(instruction_text)
        self.current_subtask_idx = 0
        self.reasoning_history = []
        self.all_subtasks_completed = False

    def _generate_scene_description(self, rgb_url: str, depth_url: str, current_subtask: str) -> str:
        """Generate scene description using both RGB and depth images."""
        prompt = (
            "You are provided with two images: the first is an RGB image from the robot's camera, and the second is a depth map where colors indicate distance (blue for close, red for far). "
            f"Describe the scene from the robot’s perspective, focusing on navigation-relevant elements related to the current task: {current_subtask}. "
            "Include specific objects (e.g., room, furniture), their positions relative to the robot (front, left, right), and their distances based on the depth map (e.g., 'a chair 1 meter in front'). "
            "Note how their positions have changed from previous scenes, indicating the robot's movement. "
            "Identify the position and direction of the target location or object if applicable (e.g., 'the arched doorway is to the right'). "
            "Highlight any obstacles directly in front of the robot and suggest possible actions to avoid them while staying on the navigation path. "
            "If navigable paths are visible (e.g., open spaces with greater depth), describe them (e.g., 'an open hallway to the left')."
            "IMPORTANT REMINDER: The robot is currently navigating in a static scanned 3D environment. It means that If you see a door in front of your, you are not able to navigate through that door since you cannot open it. It also means that you need to rely on the depth map to find navigable paths."
        )
        response = self.client.chat.completions.create(
            model="qwen-vl-max",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": rgb_url}},
                    {"type": "image_url", "image_url": {"url": depth_url}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    def _get_subtask_context(self) -> Tuple[str, str, str]:
        """Get current, previous, and next subtasks."""
        current = self.subtasks[self.current_subtask_idx]
        prev = self.subtasks[self.current_subtask_idx - 1] if self.current_subtask_idx > 0 else ""
        next_ = self.subtasks[self.current_subtask_idx + 1] if self.current_subtask_idx < len(self.subtasks) - 1 else ""
        return current, prev, next_

    def _construct_reasoning_prompt(self, instruction: str, scene_descs: List[str], history_actions: List[str]) -> str:
        current_subtask, prev_subtask, next_subtask = self._get_subtask_context()
        prompt_parts = [
            f"Navigation Task: {instruction}",
            f"Current Subtask: {current_subtask}",
            f"Previous Completed Subtask: {prev_subtask}" if prev_subtask else "",
            f"Next Subtask: {next_subtask}" if next_subtask else "",
            "\nScene Descriptions (in chronological order, earliest first):"
        ]
        for i, scene in enumerate(scene_descs[:-1]):
            prompt_parts.append(f"Previous Scene {i+1}: {scene}")
        prompt_parts.append(f"Current Scene: {scene_descs[-1]}")
        prompt_parts.append("\nReasoning Chain:")
        for i, reasoning in enumerate(self.reasoning_history[-2:]):
            prompt_parts.append(f"Previous Reasoning {i+1}: {reasoning}")
        prompt_parts.append(
            "\nInstructions:\n"
            "1. Analyze the sequence of images and their descriptions to assess the robot's movement and progress (is there any progress made?).\n"
            "2. Look for specific visual cues or landmarks that indicate whether the current subtask is completed. For example:\n"
            "   - If the subtask is to walk into a room, check if the room's features are prominently in view.\n"
            "   - If the subtask is to pass by objects, ensure those objects are no longer in the foreground.\n"
            "   - if the subtask is turn left, it typically refers to turning a 90 degree until you see a path which was originally on the left and is now in front of you."
            "3. Determine if the robot has successfully completed the current subtask based on these observations. "
            "Be conservative; only mark the subtask as completed if there is clear evidence.\n"
            "4. Decision Process for Action Selection:\n"
            "   a. Check the scene description for any obstacles directly in front of the robot (e.g., furniture like chairs). "
            "      If obstacles are present, choose 'Turn left' or 'Turn right' to avoid them, preferring the direction that "
            "      aligns with the desired path.\n"
            "   b. If no obstacles are present, check the path’s direction relative to the robot:\n"
            "      i. If the path is directly in front and leads toward the target, choose 'Move forward.'\n"
            "      ii. If the path is to the left or right, choose 'Turn left' or 'Turn right' to align with it.\n"
            "   c. If the path is aligned and no obstacles are present, but the target is not directly in front, "
            "      choose 'Turn left' or 'Turn right' to face the target.\n"
            "5. Provide step-by-step reasoning for your decision, explaining how obstacles and the target "
            "   influenced your action choice.\n\n"
            "Important Rules:\n"
            "- You MUST choose only ONE action per step.\n"
            "- Prioritize avoiding obstacles first, then aligning with the desired path, and finally facing the target.\n\n"
            "Response format:\n"
            "Subtasks Completed: [Yes/No]\n"
            "Reasoning: [your step-by-step reasoning]\n"
            "Current Action: [Must be one of: Move forward, Turn right, Turn left]"
        )
        return "\n".join(prompt_parts)

    def _parse_response(self, raw_response: str) -> str:
        lines = raw_response.split("\n")
        completion_status = "No"
        action = "Move forward"  # Default action
        reasoning = ""

        for line in lines:
            line = line.strip()
            if "Subtasks Completed:" in line:
                # Check for "yes" or "no" case-insensitively
                completion_status = "Yes" if "yes" in line.lower() else "No"
            elif "Next Action:" in line:
                # Look for action phrases in the line
                for act in self.actions:
                    if act.lower() in line.lower():
                        action = act  # Use the correctly cased action from self.actions
                        break
            elif "Next Subtask:" in line:
                # Handle case where action might be in "Next Subtask:" line
                for act in self.actions:
                    if act.lower() in line.lower():
                        action = act
                        break
            elif "Reasoning:" in line:
                reasoning = line.split(":", 1)[1].strip()

        # Fallback: if no action found, check the last non-empty line
        if action == "Move forward" and lines:
            last_line = next((line.strip().lower() for line in reversed(lines) if line.strip()), "")
            for act in self.actions:
                if act.lower() in last_line:
                    action = act
                    break

        # Update subtask progress
        if completion_status == "Yes":
            if self.current_subtask_idx < len(self.subtasks) - 1:
                self.current_subtask_idx += 1
            else:
                self.all_subtasks_completed = True

        # Store reasoning and ensure action validity
        self.reasoning_history.append(reasoning)
        if action not in self.actions:
            print("Running default action.")
            action = "Move forward"

        return action

    def _encode_image(self, image_np: np.ndarray) -> str:
        success, buffer = cv2.imencode('.jpg', image_np)
        if not success:
            raise ValueError("Could not encode image")
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    def forward(self, instruction_text: str, composite_image_np: np.ndarray, history_actions: List[str]) -> Tuple[torch.Tensor, str, str, str]:
        image_url = self._encode_image(composite_image_np)
        prompt_text = self._construct_reasoning_prompt(instruction_text, [self._generate_scene_description(image_url, self.subtasks[self.current_subtask_idx])], history_actions)
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": prompt_text}]}]
        try:
            response = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=messages,
                max_tokens=5000,
                temperature=0.1
            )
            raw_response = response.choices[0].message.content.strip()
            best_action = self._parse_response(raw_response)
        except Exception as e:
            print(f"VLM Error: {e}, using default action")
            best_action = "Move forward"
            raw_response = f"API Error: {str(e)}"
        action_map = {
            "Move forward": (0.6, 0.0, 0.0),
            "Turn right": (0.0, 0.0, -0.5),
            "Turn left": (0.0, 0.0, 0.5),
            # "Stop moving": (0.0, 0.0, 0.0)
        }
        return torch.tensor(action_map[best_action], device=self.device).unsqueeze(0), best_action, prompt_text, raw_response

class LLMReasoner:
    """Handles instruction decomposition using an LLM."""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

    def decompose_instruction(self, instruction: str) -> List[str]:
        """Decompose navigation instruction into subtasks."""
        prompt = (
            "Below is an example of how to decompose a navigation instruction into subtasks:\n\n"
            "Instruction: Go to the kitchen and grab a glass of water.\n"
            "Output format: Numbered list of subtasks without additional text\n"
            "1. Go to the kitchen\n"
            "2. Grab a glass of water\n\n"
            "Now, decompose the following navigation instruction into sequential subtasks in the same format:\n\n"
            f"Instruction: {instruction}\n"
        )
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            response_content = response.choices[0].message.content.strip()
            lines = response_content.split("\n")
            subtasks = [line.split(". ", 1)[1] for line in lines if ". " in line]
            return subtasks + ["Goal Reached"] if subtasks else ["Goal Reached"]
        except Exception as e:
            print(f"Error in decompose_instruction: {e}")
            return ["Goal Reached"]