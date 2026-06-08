import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from PIL import Image

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. MOCK DATA GENERATOR (For Demonstration)
# ==========================================
def generate_mock_dataset(num_samples=100, img_dir="mock_house_images"):
    """Generates synthetic tabular data and dummy image assets."""
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        
    data = {
        "id": [f"house_{i}" for i in range(num_samples)],
        "bedrooms": np.random.randint(1, 6, size=num_samples),
        "bathrooms": np.random.randint(1, 4, size=num_samples),
        "sqft": np.random.randint(800, 4500, size=num_samples),
        "age": np.random.randint(0, 80, size=num_samples),
        "price": np.random.randint(150000, 1200000, size=num_samples)
    }
    
    # Save dummy visual arrays as JPEG assets
    for i in range(num_samples):
        dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(dummy_array)
        img.save(os.path.join(img_dir, f"house_{i}.jpg"))
        
    df = pd.DataFrame(data)
    df.to_csv("mock_housing_data.csv", index=False)
    print("Mock dataset generated successfully.")

# Execute generation
generate_mock_dataset(num_samples=120)


# ==========================================
# 2. MULTIMODAL PYTORCH DATASET
# ==========================================
class MultimodalHousingDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_train=True, scaler=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Split features vs targets
        self.labels = self.df["price"].values.astype(np.float32)
        self.tabular_features = self.df[["bedrooms", "bathrooms", "sqft", "age"]].values.astype(np.float32)
        
        # Scale tabular data
        if is_train:
            self.scaler = StandardScaler()
            self.tabular_features = self.scaler.fit_transform(self.tabular_features)
        else:
            self.scaler = scaler
            self.tabular_features = self.scaler.transform(self.tabular_features)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Fetch Tabular Vector
        tab_vector = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        
        # Fetch Image
        img_id = self.df.iloc[idx]["id"]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Target
        price = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return tab_vector, image, price


# ==========================================
# 3. MULTIMODAL NETWORK ARCHITECTURE
# ==========================================
class MultimodalHousingModel(nn.Module):
    def __init__(self, num_tabular_features):
        super(MultimodalHousingModel, self).__init__()
        
        # Modality A: CNN Feature Extractor (Pretrained EfficientNet backbone)
        # Using weights parameter as per modern torchvision API
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_cnn_features = self.cnn.classifier[1].in_features
        # Strip out standard classification head, replace with feature pooling layer
        self.cnn.classifier = nn.Identity() 
        
        # Modality B: Tabular MLP Block
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )
        
        # Late Fusion Integration Head
        # Combines cnn feature size (1280) + tabular size (32)
        combined_features_dim = num_cnn_features + 32
        
        self.regressor = nn.Sequential(
            nn.Linear(combined_features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single linear value out for continuous regression target
        )

    def forward(self, tab_data, img_data):
        # Forward pass on image through deep backbone
        img_features = self.cnn(img_data) # Shape: [batch_size, 1280]
        
        # Forward pass on structured data
        tab_features = self.tabular_mlp(tab_data) # Shape: [batch_size, 32]
        
        # Concatenate features horizontally along column dimension
        fused_features = torch.cat((img_features, tab_features), dim=1)
        
        # Final pricing prediction
        output = self.regressor(fused_features)
        return output.squeeze(-1)


# ==========================================
# 4. TRAINING & EVALUATION PIPELINE
# ==========================================
def main():
    # Setup Image Transforms
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize and Partition Datasets
    # First, split the metadata CSV into train and validation sets
    raw_df = pd.read_csv("mock_housing_data.csv")
    train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=42)
    
    train_df.to_csv("train_split.csv", index=False)
    val_df.to_csv("val_split.csv", index=False)
    
    train_dataset = MultimodalHousingDataset("train_split.csv", "mock_house_images", transform=img_transforms, is_train=True)
    val_dataset = MultimodalHousingDataset("val_split.csv", "mock_house_images", transform=img_transforms, is_train=False, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Device Assignment (GPU fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline execution on: {device}")
    
    model = MultimodalHousingModel(num_tabular_features=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training Loop
    epochs = 15
    print("Beginning Training Strategy...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for tab_data, img_data, targets in train_loader:
            tab_data, img_data, targets = tab_data.to(device), img_data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(tab_data, img_data)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * tab_data.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss (MSE): {epoch_loss:.2f}")

    # Validation & Final Performance Evaluation (MAE and RMSE)
    model.eval()
    absolute_errors = []
    squared_errors = []
    
    with torch.no_grad():
        for tab_data, img_data, targets in val_loader:
            tab_data, img_data, targets = tab_data.to(device), img_data.to(device), targets.to(device)
            predictions = model(tab_data, img_data)
            
            # Record structural metric performance array elements
            abs_err = torch.abs(predictions - targets)
            sq_err = (predictions - targets) ** 2
            
            absolute_errors.extend(abs_err.cpu().numpy())
            squared_errors.extend(sq_err.cpu().numpy())
            
    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(squared_errors))
    
    print("\n==================================")
    print("   FINAL MULTIMODAL EVALUATION    ")
    print("==================================")
    print(f"Mean Absolute Error (MAE):  ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")


if __name__ == "__main__":
    main()