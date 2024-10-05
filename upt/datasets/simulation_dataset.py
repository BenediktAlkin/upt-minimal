import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    def __init__(
            self,
            root,
            # how many input points to sample
            num_inputs,
            # how many output points to sample
            num_outputs,
            # train or rollout mode
            # - train: next timestep prediction
            # - rollout: return all timesteps for visualization
            mode,
    ):
        super().__init__()
        root = Path(root).expanduser()
        self.root = root
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mode = mode
        # discover simulations
        self.case_names = list(sorted(os.listdir(root)))
        self.num_timesteps = len(
            [
                fname for fname in os.listdir(root / self.case_names[0])
                if fname.endswith("_mesh.th")
            ],
        )
        self.mean = torch.tensor([0.0152587890625, -1.7881393432617188e-06, 0.0003612041473388672])
        self.std = torch.tensor([0.0233612060546875, 0.0184173583984375, 0.0019378662109375])

    def __len__(self):
        if self.mode == "train":
            # first timestep cant be predicted
            return len(self.case_names) * (self.num_timesteps - 1)
        elif self.mode == "rollout":
            return len(self.case_names)
        else:
            raise NotImplementedError(f"invalid mode: '{self.mode}'")

    def __getitem__(self, idx):
        if self.mode == "train":
            # return t and t + 1
            case_idx = idx // (self.num_timesteps - 1)
            timestep = idx % (self.num_timesteps - 1)
            case_uri = self.root / self.case_names[case_idx]
            x = torch.load(case_uri / "x.th", weights_only=True).float()
            y = torch.load(case_uri / "y.th", weights_only=True).float()
            pos = torch.stack([x, y], dim=1)
            input_pos = pos
            output_pos = pos
            input_feat = torch.load(case_uri / f"{timestep:08d}_mesh.th", weights_only=True).float().T
            output_feat = torch.load(case_uri / f"{timestep + 1:08d}_mesh.th", weights_only=True).float().T
            # subsample inputs
            if self.num_inputs != float("inf"):
                input_perm = torch.randperm(len(input_feat))[:self.num_inputs]
                input_feat = input_feat[input_perm]
                input_pos = input_pos[input_perm]
            # subsample outputs
            if self.num_outputs != float("inf"):
                output_perm = torch.randperm(len(output_feat))[:self.num_outputs]
                output_feat = output_feat[output_perm]
                output_pos = output_pos[output_perm]
        elif self.mode == "rollout":
            # return all timesteps
            assert self.num_inputs == float("inf")
            assert self.num_outputs == float("inf")
            timestep = 0
            case_uri = self.root / self.case_names[idx]
            x = torch.load(case_uri / "x.th", weights_only=True).float()
            y = torch.load(case_uri / "y.th", weights_only=True).float()
            pos = torch.stack([x, y], dim=1)
            input_pos = pos
            output_pos = pos
            data = [
                torch.load(case_uri / f"{i:08d}_mesh.th", weights_only=True).float().T
                for i in range(self.num_timesteps)
            ]
            input_feat = data[0]
            output_feat = data[1:]
        else:
            raise NotImplementedError

        # scale x positions from [-0.5, 0.5] to [0-200]
        # scale y positions from [-0.5, 1] to [0-300]
        input_pos = (input_pos + 0.5) * 200
        output_pos = (output_pos + 0.5) * 200

        # normalize
        input_feat -= self.mean.unsqueeze(0)
        input_feat /= self.std.unsqueeze(0)
        if isinstance(output_feat, list):
            for i in range(len(output_feat)):
                output_feat[i] -= self.mean.unsqueeze(0)
                output_feat[i] /= self.std.unsqueeze(0)
        else:
            output_feat -= self.mean.unsqueeze(0)
            output_feat /= self.std.unsqueeze(0)

        return dict(
            index=idx,
            input_feat=input_feat,
            input_pos=input_pos,
            output_feat=output_feat,
            output_pos=output_pos,
            timestep=timestep,
        )
