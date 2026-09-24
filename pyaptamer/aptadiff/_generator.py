"""AptaDiff engine: trains the diffusion model and generates sequences."""

__author__ = ["aditi-dsi"]
__all__ = ["AptaDiffGenerator"]

from sklearn.base import BaseEstimator


class AptaDiffGenerator(BaseEstimator):
    def __init__(
        self,
        num_classes=4,
        num_timesteps=1000,
        dim=512,
        depth=12,
        n_blocks=1,
        heads=16,
        attn_layer_dropout=0.0,
        n_local_attn_heads=8,
        local_attn_window_size=1,
        transformer_type="native",
        max_epochs=1000,
        batch_size=32,
        lr=1e-4,
        betas=(0.9, 0.999),
        gamma=0.99,
        accumulate_grad_batches=1,
        validation_fraction=0.1,
        random_state=None,
        accelerator="auto",
        verbose=0,
    ):
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.dim = dim
        self.depth = depth
        self.n_blocks = n_blocks
        self.heads = heads
        self.attn_layer_dropout = attn_layer_dropout
        self.n_local_attn_heads = n_local_attn_heads
        self.local_attn_window_size = local_attn_window_size
        self.transformer_type = transformer_type
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.accumulate_grad_batches = accumulate_grad_batches
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.accelerator = accelerator
        self.verbose = verbose

    def _to_onehot(self, X):
        pass

    def _build(self, seq_len, embed_size):
        pass

    def fit(self, X, y):
        pass

    def sample(self, y, random_state=None):
        pass

    def score(self, X, y):
        pass

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags
