from functools import partial

import einops
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection, Sequential
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock, DitBlock
from kappamodules.vit import VitBlock
from torch import nn
from torch_geometric.utils import unbatch


class DecoderPerceiver(nn.Module):
    def __init__(
            self,
            input_dim,
            ndim,
            dim,
            depth,
            num_heads,
            perc_dim=None,
            perc_num_heads=None,
            cond_dim=None,
            init_weights="truncnormal002",
            **kwargs,
    ):
        super().__init__(**kwargs)
        perc_dim = perc_dim or dim
        perc_num_heads = perc_num_heads or num_heads
        self.input_dim = input_dim
        self.ndim = ndim
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.cond_dim = cond_dim
        self.init_weights = init_weights

        # input projection
        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights, optional=True)

        # blocks
        if cond_dim is None:
            block_ctor = VitBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
        self.blocks = Sequential(
            *[
                block_ctor(
                    dim=dim,
                    num_heads=num_heads,
                    init_weights=init_weights,
                )
                for _ in range(depth)
            ],
        )

        # prepare perceiver
        self.pos_embed = ContinuousSincosEmbed(
            dim=perc_dim,
            ndim=ndim,
        )
        if cond_dim is None:
            block_ctor = PerceiverBlock
        else:
            block_ctor = partial(DitPerceiverBlock, cond_dim=cond_dim)

        # decoder
        self.query_proj = nn.Sequential(
            LinearProjection(perc_dim, perc_dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim, perc_dim, init_weights=init_weights),
        )
        self.perc = block_ctor(dim=perc_dim, kv_dim=dim, num_heads=perc_num_heads, init_weights=init_weights)
        self.pred = nn.Sequential(
            nn.LayerNorm(perc_dim, eps=1e-6),
            LinearProjection(perc_dim, 3, init_weights=init_weights),
        )

    def forward(
            self,
            x,
            output_pos,
            unbatch_idx=None,
            unbatch_select=None,
            condition=None,
    ):
        # check inputs
        assert x.ndim == 3, "expected shape (batch_size, num_latent_tokens, dim)"
        assert output_pos.ndim == 3, "expected shape (batch_size, num_outputs, dim) num_outputs might be padded"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # input projection
        x = self.input_proj(x)

        # apply blocks
        x = self.blocks(x, **cond_kwargs)

        # create query
        query = self.pos_embed(output_pos)
        query = self.query_proj(query)

        x = self.perc(q=query, kv=x, **cond_kwargs)
        x = self.pred(x)
        # dense tensor (batch_size, max_num_points, dim) -> sparse tensor (batch_size * num_points, dim)
        hat = einops.rearrange(x, "batch_size max_num_points dim -> (batch_size max_num_points) dim")
        if unbatch_idx is None:
            # with batch_size=1 no padding is needed --> no unbatching is needed
            assert len(output_pos) == 1
        else:
            unbatched = unbatch(hat, batch=unbatch_idx)
            hat = torch.concat([unbatched[i] for i in unbatch_select])

        return hat
