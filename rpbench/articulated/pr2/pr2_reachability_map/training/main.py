import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm
from plainmp.robot_spec import PR2LarmSpec, PR2RarmSpec
from scipy.spatial import KDTree
from skrobot.coordinates.math import rpy2quaternion, wxyz2xyzw
from torch.utils.data import DataLoader, Dataset, random_split

from rpbench.articulated.pr2.pr2_reachability_map.model import FCN, Domain


def create_dataset(is_rarm: bool, n_sample: int):
    if is_rarm:
        spec = PR2RarmSpec()
        frame = "r_gripper_tool_frame"
    else:
        spec = PR2LarmSpec()
        frame = "l_gripper_tool_frame"
    kin = spec.get_kin()
    torso_ids = kin.get_joint_ids(["torso_lift_joint"])
    kin.set_joint_positions(torso_ids, np.array([0.0]))

    lb, ub = spec.angle_bounds()
    ctrl_ids = kin.get_joint_ids(spec.control_joint_names)
    dataset = np.zeros(
        (2 * n_sample, 7), dtype=np.float32
    )  # 2 for two different values of quaternion for the same rotation

    for i in tqdm.tqdm(range(n_sample)):
        q = np.random.uniform(lb, ub)
        kin.set_joint_positions(ctrl_ids, q)
        pose = kin.debug_get_link_pose(frame)
        dataset[2 * i] = pose
        dataset[2 * i + 1, :3] = pose[:3]
        dataset[2 * i + 1, 3:] = pose[3:] * -1
    return dataset


def create_dense_enough_dataset(is_rarm: bool, threshold: float) -> np.ndarray:
    n_sample_each = 1000_000
    n_sample_test = 10_000
    dist_threshold = threshold
    dataset = np.zeros((0, 7), dtype=np.float32)

    while True:
        dataset_add = create_dataset(is_rarm, n_sample_each)
        dataset = np.vstack((dataset, dataset_add))
        kdtree = KDTree(dataset)
        testset = create_dataset(is_rarm, n_sample_test)
        print("testing..")
        dist, _ = kdtree.query(testset, k=1)
        dist_quantile = np.quantile(dist, 0.99)
        print(f"99% quantile of distance: {dist_quantile}")
        if dist_quantile < dist_threshold:
            break
    return dataset


class ReachableScoreDataset(Dataset):
    def __init__(self, tree: KDTree):
        domain = Domain()
        n_sample = 1000_000
        X = np.zeros((n_sample, 7), dtype=np.float32)
        Y = np.zeros((n_sample,), dtype=np.float32)
        for i in tqdm.tqdm(range(n_sample), desc="Creating ReachableScoreDataset"):
            x = domain.sample_point()
            pos, rpy = x[:3], x[3:]
            quat_wxyz = rpy2quaternion(rpy[::-1])
            quat_xyzw = wxyz2xyzw(quat_wxyz)
            x = np.concatenate([pos, quat_xyzw])
            X[i] = x
            Y[i], _ = tree.query(x, k=1)
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="rarm", choices=["rarm", "larm"])
    args = parser.parse_args()

    print(f"Train model for {args.arm}...")
    dataset = create_dense_enough_dataset(args.arm == "rarm", threshold=0.15)
    tree = KDTree(dataset)
    reach_dataset = ReachableScoreDataset(tree)

    dataset_size = len(reach_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(reach_dataset, [train_size, val_size])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FCN(input_dim=7, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10000
    best_val_loss = float("inf")
    epoch_last_update = -1
    early_stopping_patience = 100

    pretrained_dir = Path(pr2_reachability_map.__file__).parent / "pretrained"
    best_model_path = pretrained_dir / f"best_model_{args.arm}.pth"

    for epoch in range(num_epochs):
        if epoch - epoch_last_update > early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Training phase
        model.train()
        running_train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            # Ensure Y has shape [batch_size, 1] for MSELoss
            Y_batch = Y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / train_size

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device).unsqueeze(1)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                running_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = running_val_loss / val_size

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = copy.deepcopy(model).to("cpu")
            torch.save(best_model.state_dict(), best_model_path)
            print("Saved new best model.")
            epoch_last_update = epoch
