from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from skrobot.coordinates import Coordinates, Transform
from skrobot.coordinates.math import (
    matrix2quaternion,
    quaternion2matrix,
    rpy_matrix,
    wxyz2xyzw,
    xyzw2wxyz,
)


@dataclass
class Domain:
    lb: np.ndarray = np.array([-0.5, -1.5, 0.0, -np.pi, -np.pi, -np.pi])
    ub: np.ndarray = np.array([1.5, 1.5, 2.0, np.pi, np.pi, np.pi])

    def sample_point(self) -> np.ndarray:
        return np.random.uniform(self.lb, self.ub)


class FCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


@dataclass
class ReachabilityClassifier:
    fcn: FCN
    threshold: float = 0.1
    torso_position: float = 0.0
    pr2_pose: Optional[np.ndarray] = None
    domain: Domain = Domain()
    tf_world_to_pr2: Optional[Transform] = Transform(np.zeros(3), np.eye(3))

    def set_base_pose(self, pr2_pose: np.ndarray):
        x, y, yaw = pr2_pose
        tf_pr2_to_world = Transform([x, y, 0.0], rpy_matrix(yaw, 0, 0))
        tf_world_to_pr2 = tf_pr2_to_world.inverse_transformation()
        self.tf_world_to_pr2 = tf_world_to_pr2

    def predict(self, x: Union[np.ndarray, Transform, Coordinates]) -> bool:
        if isinstance(x, np.ndarray):
            pos = x[:3]
            rot = x[3:]
            if len(rot) == 3:  # rpy
                mat = rpy_matrix(*rot[::-1])
            else:
                mat = quaternion2matrix(xyzw2wxyz(rot))
            tf_pose_to_world = Transform(pos, mat)
        elif isinstance(x, Coordinates):
            tf_pose_to_world = Transform(x.translation, x.rotation)
        else:
            tf_pose_to_world = x

        if np.any(tf_pose_to_world.translation < self.domain.lb[:3]) or np.any(
            tf_pose_to_world.translation > self.domain.ub[:3]
        ):
            return False
        assert isinstance(tf_pose_to_world, Transform)
        tf_pose_to_pr2 = tf_pose_to_world * self.tf_world_to_pr2
        inp = np.hstack(
            [tf_pose_to_pr2.translation, wxyz2xyzw(matrix2quaternion(tf_pose_to_pr2.rotation))]
        )
        inp[2] -= self.torso_position
        inp_ten = torch.from_numpy(inp).float().unsqueeze(0)
        ret = self.fcn(inp_ten).item()
        return ret < self.threshold


def get_model_path(arm: Literal["rarm", "larm"]):
    pretrained_path = (Path(__file__).parent / "pretrained").expanduser()
    if arm == "rarm":
        model_path = pretrained_path / "best_model_rarm.pth"
    elif arm == "larm":
        model_path = pretrained_path / "best_model_larm.pth"
    else:
        raise ValueError(f"Invalid arm {arm}")
    return model_path


def load_classifier(arm: Literal["rarm", "larm"]) -> ReachabilityClassifier:
    model = FCN(7, 1)
    model_path = get_model_path(arm)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.cpu()
    dummy_input = torch.randn(1, 7)
    traced = torch.jit.trace(model, (dummy_input,))
    optimized = torch.jit.optimize_for_inference(traced)
    return ReachabilityClassifier(optimized)


if __name__ == "__main__":
    model = load_classifier("rarm")
    domain = Domain()
    p = np.array([0.6, -0.3, 0.8, 0.0, 0.0, 0.0])
    print(model.predict(p))
