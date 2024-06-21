import torch

from upt import UPT, EncoderSupernodes, DecoderPerceiver, Approximator


def main():
    # create a single data point
    torch.manual_seed(0)
    num_inputs = 8
    num_outputs = 16
    pos_scale = 1000
    ndim = 3
    input_feat = torch.randn(num_inputs, 4)
    input_pos = torch.rand(num_inputs, ndim) * pos_scale
    supernode_idxs = torch.tensor([2, 5])
    batch_idx = torch.zeros(num_inputs, dtype=torch.long)
    output_pos = torch.rand(1, num_outputs, ndim) * pos_scale

    # create model
    model = UPT(
        encoder=EncoderSupernodes(
            input_dim=input_feat.size(1),
            ndim=ndim,
            radius=5,
            max_degree=32,
            gnn_dim=128,
            enc_dim=128,
            enc_depth=4,
            enc_num_heads=2,
        ),
        approximator=Approximator(
            input_dim=128,
            dim=128,
            num_heads=2,
            depth=4,
        ),
        decoder=DecoderPerceiver(
            input_dim=128,
            dim=128,
            num_heads=2,
            depth=4,
            ndim=ndim,
        ),
    )

    pred = model(
        input_feat=input_feat,
        input_pos=input_pos,
        supernode_idxs=supernode_idxs,
        output_pos=output_pos,
        batch_idx=batch_idx,
    )
    print(pred.shape)


if __name__ == "__main__":
    main()
