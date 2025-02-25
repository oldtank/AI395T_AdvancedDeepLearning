import abc

import torch


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        # Positional embedding
        self.positional_embedding = torch.nn.Embedding(1024, d_latent)  # Assuming max seq_len = 1024

        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=8, dim_feedforward=4 * d_latent, activation='relu'),
            num_layers=6
        )
        self.linear = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, height, width = x.shape
        sequence_length = height * width
        
        # flatten
        x_flatten = x.view(batch_size, sequence_length)

        # Shift input implicitly by one position
        x_shifted = torch.zeros_like(x_flatten)
        x_shifted[:, 1:] = x_flatten[:, :-1]

        x_embedding = self.embedding(x_shifted)

        # Positional embedding
        positions = torch.arange(sequence_length, device=x.device).unsqueeze(0).expand(batch_size, sequence_length)
        pos_emb = self.positional_embedding(positions)

        mask = torch.nn.Transformer.generate_square_subsequent_mask(sequence_length).to(x.device)

        x_embedding_combined = x_embedding + pos_emb

        # Prepare input for the transformer: (seq_len, B, d_latent)
        x_embedding_combined = x_embedding_combined.permute(1, 0, 2)
        
        # Transformer encoder
        output = self.transformer_encoder(x_embedding_combined, mask=mask)  # (seq_len, batch_size, d_latent)

        # Linear layer
        output = self.linear(output)

        probs = torch.softmax(output, dim=-1)

        # Reshape to image dimensions
        probs = probs.permute(1, 0, 2).view(batch_size, height, width, self.n_tokens)

        # # Softmax for probabilities
        # output_probs = torch.softmax(output, dim=-1)

        return probs, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        raise NotImplementedError()
