

class UPTCollator:
    def __init__(self, num_supernodes, deterministic):
        self.num_supernodes = num_supernodes
        self.deterministic = deterministic

    def __call__(self, batch):
        return batch