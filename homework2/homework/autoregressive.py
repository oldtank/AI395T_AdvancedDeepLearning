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
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_latent, nhead=2),
            3
        )
        self.linear = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, height, width = x.shape
        sequence_length = height * width
        
        # flatten
        x_flatten = x.view(batch_size, sequence_length)

        x_embedding = self.embedding(x_flatten)

        zeros = torch.zeros(batch_size, 1, self.d_latent, device=x.device)
        embeddding_shifted = torch.cat([zeros, x_embedding[:, :-1, :]], dim=1) # (batch_size, seq_len, d_latent)

        # Transpose for transformer encoder
        embedding_shifted_transposed = embeddding_shifted.transpose(0, 1)  # (sequence_length, batch_size, d_latent)

        mask = torch.nn.Transformer.generate_square_subsequent_mask(sequence_length).to(x.device)
        
        # Transformer encoder
        output = self.transformer_encoder(embedding_shifted_transposed, mask=mask)  # (seq_len, batch_size, d_latent)

        # Linear layer
        output = self.linear(output).transpose(0, 1)  # (batch_size, seq_len, n_tokens)

        # Reshape to image dimensions
        output = output.view(batch_size, height, width, self.n_tokens)

        # Softmax for probabilities
        output_probs = torch.softmax(output, dim=-1)

        return output_probs, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        raise NotImplementedError()
