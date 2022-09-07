import sys
sys.path.append("../scop_classification_by_ESM")
import torch

class Config(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Config, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        self.task="SF"
        self.max_len=512
        self.dropout=0.5
        self.init_lr=1e-5 #1e-4
        self.n_epochs=400
        self.batch_size=32
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # "cpu"#
        
    
    def get_model_name(self, prefix="ESM1b") -> str:
        out_filename = f"ESM1b_{self.task}_{self.max_len}_{self.dropout}_{self.init_lr}_{self.n_epochs}_{self.batch_size}_{self.device}"
        return out_filename