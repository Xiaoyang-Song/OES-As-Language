import torch
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, embed_dim, num_heads, num_layers, ff_dim, dropout):
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len

        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(embed_dim, 6)  # Output vector of dimension 6

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # Apply embedding
        x = self.embedding(x)

        # Add positional encoding
        x += self.positional_encoding

        # Permute to match Transformer input shape (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Take the mean across the sequence dimension (Pooling)
        x = x.mean(dim=0)

        # Output layer
        output = self.fc_out(x)

        return output

class CNNGenerator(nn.Module):
    def __init__(self, input_dim, output_channels, img_size):
        super(CNNGenerator, self).__init__()

        self.input_dim = input_dim
        self.output_channels = output_channels
        self.img_size = img_size
        self.init_size = img_size // 4  # Initial size after first upsampling
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128 * self.init_size * self.init_size),
            nn.ReLU()
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),

            # First upsampling layer
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # Second upsampling layer
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Output layer
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x):
        # Fully connected layer to reshape into 2D feature maps
        x = self.fc(x)
        x = x.view(x.size(0), 128, self.init_size, self.init_size)

        # Pass through convolutional blocks
        img = self.conv_blocks(x)
        return img

class TransformerToImage(nn.Module):
    def __init__(self, transformer, generator):
        super(TransformerToImage, self).__init__()
        self.transformer = transformer
        self.generator = generator

    def forward(self, x):
        # Pass through transformer to get a vector
        vector = self.transformer(x)

        # Pass the vector through the generator to get an image
        img = self.generator(vector)
        return img


if __name__ == '__main__':
    # Model configuration for Transformer
    input_dim = 1      # Feature size per timestep
    seq_len = 16       # Length of the sequence
    embed_dim = 32     # Embedding dimension
    num_heads = 4      # Number of attention heads
    num_layers = 2     # Number of transformer layers
    ff_dim = 64        # Feedforward network dimension
    dropout = 0.1      # Dropout rate

    # Model configuration for CNN Generator
    generator_input_dim = 6
    output_channels = 1     # Output image channels (e.g., 3 for RGB images)
    img_size = 32           # Output image size (32x32)

    # Initialize models
    transformer = TransformerModel(input_dim, seq_len, embed_dim, num_heads, num_layers, ff_dim, dropout)
    generator = CNNGenerator(generator_input_dim, output_channels, img_size)

    # Combine models
    model = TransformerToImage(transformer, generator)

    # Example input: batch of 8 sequences, each with shape (16, 1)
    x = torch.rand(8, seq_len, input_dim)

    # Forward pass
    output_images = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output_images.shape)
