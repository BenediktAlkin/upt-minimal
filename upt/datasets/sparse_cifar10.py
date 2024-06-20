import einops
import torch
from torchvision.datasets import CIFAR10


class SparseCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False, num_inputs=256):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )
        assert num_inputs <= 1024, "CIFAR10 only has 1024 pixels, use less or equal 1024 num_inputs"
        self.num_inputs = num_inputs
        # CIFAR has 32x32 pixels
        # output_pos will be a tensor of shape (32 * 32, 2) with and will contain x and y indices
        # output_pos[0] = [0, 0]
        # output_pos[1] = [0, 1]
        # output_pos[2] = [0, 2]
        # ...
        # output_pos[32] = [1, 0]
        # output_pos[1024] = [31, 31]
        self.output_pos = einops.rearrange(
            torch.stack(torch.meshgrid([torch.arange(32), torch.arange(32)], indexing="ij")),
            "ndim height width -> (height width) ndim",
        )
        # convert output_pos from [0, 31] to [0, 1000] for better behavior with sin-cos pos embeddings
        self.output_pos = self.output_pos / 31 * 1000

    def __getitem__(self, idx):
        image, y = super().__getitem__(idx)
        assert image.shape == (3, 32, 32)
        # reshape image to sparse tensor
        x = einops.rearrange(image, "dim height width -> (height width) dim")
        pos = self.output_pos.clone()

        # subsample random pixels
        if self.num_inputs < 1024:
            if self.train:
                rng = None
            else:
                rng = torch.Generator().manual_seed(idx)
            perm = torch.randperm(len(x), generator=rng)[:self.num_inputs]
            x = x[perm]
            pos = pos[perm].clone()

        return dict(
            index=idx,
            input_feat=x,
            input_pos=pos,
            target_class=y,
            target_image=image,
        )
