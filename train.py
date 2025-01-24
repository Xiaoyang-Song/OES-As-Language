from model import *
import os
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model configuration for Transformer
input_dim = 1      # Feature size per timestep
seq_len = 16       # Length of the sequence
embed_dim = 64     # Embedding dimension
num_heads = 4      # Number of attention heads
num_layers = 8     # Number of transformer layers
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
model = TransformerToImage(transformer, generator).to(DEVICE)

# Data
data_dir = os.path.join('Data', 'simulation-circle-crack')
# Load training
ct_tri = torch.load(os.path.join(data_dir, 'ct.pt'))
sp_tri = torch.load(os.path.join(data_dir, 'sp_2fc.pt'))
print("CT shape: ", ct_tri.shape)
print("SP shape: ", sp_tri.shape)
ct_test = torch.load(os.path.join(data_dir, 'ct-test.pt'))
sp_test = torch.load(os.path.join(data_dir, 'sp_2fc-test.pt'))
print("CT test shape: ", ct_tri.shape)
print("SP test shape: ", sp_tri.shape)

train_data = list(zip(sp_tri, ct_tri))
test_data = list(zip(sp_test, ct_test))
train_loader = torch.utils.dataloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = torch.utils.dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)


num_epochs=1000
criterion = nn.MSELoss()  # Reconstruction loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0.0
    model.train()
    for sequences, images in train_loader:
        sequences, images = sequences.to(DEVICE), images.to(DEVICE)

        # Forward pass
        outputs = model(sequences.unsqueeze(-1))

        # Compute loss
        loss = criterion(outputs, images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    with torch.no_grad():
        epoch_loss_test = 0
        for sequences, images in test_loader:
            sequences, images = sequences.to(DEVICE), images.to(DEVICE)

            # Forward pass
            outputs = model(sequences.unsqueeze(-1))

            # Compute loss
            loss = criterion(outputs, images)

            epoch_loss_test += loss.item()

        avg_loss = epoch_loss_test / len(test_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

torch.save(model, os.path.join('ckpt', 'model.pt'))