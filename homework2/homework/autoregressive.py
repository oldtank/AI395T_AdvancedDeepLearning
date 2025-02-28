import abc

import torch
import torch.nn.functional as F


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
        self.embedding = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_latent)
        # Positional embedding
        # self.positional_embedding = torch.nn.Embedding(1024, d_latent)  # Assuming max seq_len = 1024

        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=8, dim_feedforward=4 * d_latent, batch_first=True,
                                                                 dropout=0.1)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer = self.transformer_layer,
            num_layers=2
        )
        self.linear = torch.nn.Linear(d_latent, n_tokens)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, height, width = x.shape
        sequence_length = height * width
        
        # flatten
        x_flatten = x.view(batch_size, sequence_length)
        # shift
        x_shifted = x_flatten[:, :-1]  # All tokens except the last one
        padding = torch.zeros((batch_size, 1), dtype=x.dtype, device=x.device)
        x_padded = torch.cat([padding, x_shifted], dim=1)
        # token embedding
        # print("before embedding shape: ", x_padded.shape)
        x_embedding = self.embedding(x_padded)
        # print("embedding shape: ", x_embedding.shape)

        mask = torch.nn.Transformer.generate_square_subsequent_mask(sequence_length).to(x.device)
        
        # Transformer encoder
        transformer_output = self.transformer_encoder(x_embedding, mask=mask)  # (seq_len, batch_size, d_latent)

        # print("transformer output shape: ", transformer_output.shape)
        # Linear layer
        logits = self.linear(transformer_output)
        # print("logits shape: ", logits.shape)
        
        # # Softmax for probabilities
        # probabilities = F.softmax(self.relu(logits), dim=-1)
        probabilities_reshaped = logits.view(batch_size, height, width, self.n_tokens)

        # print(probabilities_reshaped)

        return probabilities_reshaped, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        seq_len = h * w
        # Start with an empty image (e.g., all zeros or a special start token)
        x = torch.randint(0, 1024, (B, h, w), dtype=torch.long, device=device)
        generated_tokens = torch.zeros(B, h, w, dtype=torch.long, device=device)
        
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for i in range(h*w):
                row = i // w  # Compute row index
                col = i % w   # Compute column index
                
                logits, _ = self.forward(x)
                probs = logits[:, row, col, :]
                # print(f"row {row} column {col} probs {probs}")
                
                next_token = torch.argmax(probs, dim=-1)
                print(next_token)
                generated_tokens[:, row, col] = next_token

                x[:, row, col] = next_token
        return generated_tokens
