import os
import clip
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from finematch_dataset import FineMatchDataset 
from utils import FastDataLoader 
import neptune

class ClassificationHead(nn.Module):
    def __init__(self, clipmodel):
        super(ClassificationHead, self).__init__()
        self.clipmodel = clipmodel
        self.projection_size = self.clipmodel.text_projection.shape[1]
       
        # Define a simple MLP
        self.layers = []
        self.num_layers = 3
        self.hidden_size = 128
       
        input_size = self.projection_size
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(input_size, self.hidden_size))
            self.layers.append(nn.ReLU())  # Add a non-linear activation
            input_size = self.hidden_size
        self.layers.append(nn.Linear(input_size, 1))  # Final layer for binary classification
        self.layers.append(nn.Sigmoid())  # Output activation

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, similarity):
        # Ensure the similarity input is in the correct dtype
        similarity = similarity.type(self.mlp[0].weight.dtype)  # Match the first layer's weight dtype
        return self.mlp(similarity)
    

if __name__ == "__main__":

    run = neptune.init_run(
    project="bidur/noisy-correspondence",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YTdmODdjMy1iNzZlLTRhMjMtYTU2ZS1mYmQyNDU0YmJmNDIifQ==",
)  # your credentials    


    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print (device)
    model, preprocess = clip.load('ViT-L/14', device)


    # File paths
    jsonl_file = "data_labels/FineMatch_test.jsonl"  # Replace with the path to your processed JSONL file
    image_folder = "D:/finematch/images/FineMatch"  # Replace with the path to your image folder


    # Create the dataset
    dataset = FineMatchDataset(jsonl_file, image_folder, transform=preprocess)

    # Create the dataloader
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True,num_workers=4, pin_memory=True)
    dataloader = FastDataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # Instantiate the full model
    classification_head = ClassificationHead(model).to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = torch.optim.Adam(classification_head.parameters(), lr=1e-2) # Optimizer

  
    num_epochs = 50

    for epoch in range(num_epochs):

        # Initialize epoch loss and accuracy
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Process a single batch
        for images, queries, labels, labels_type, targets in dataloader:
            image_inputs = images.to(device)  # Image batch
            text_inputs = clip.tokenize(labels).to(device)  # Text inputs
            targets = targets.to(device,dtype=torch.float)  # Target indices
           
            with torch.no_grad():
            
                # Extract embeddings
                image_features = model.encode_image(image_inputs)
                text_features = model.encode_text(text_inputs)

                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            
            # Compute similarity
            similarity = image_features * text_features

            # Normalize the similarity vector to range [0, 1]
            # similarity_min = similarity.min(dim=-1, keepdim=True)[0]
            # similarity_max = similarity.max(dim=-1, keepdim=True)[0]
            # similarity = (similarity - similarity_min) / (similarity_max - similarity_min + 1e-8)

        
            # Forward pass
            outputs = classification_head(similarity).squeeze(1)
            
            loss = criterion(outputs, targets)

            # Calculate accuracy
            predictions = (outputs > 0.5).float()  # Convert probabilities to binary labels
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()


        # Compute average loss and accuracy for the epoch
        avg_loss = epoch_loss / len(dataloader)
        accuracy = (correct_predictions / total_samples) * 100

        # Log loss and accuracy for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        run["train/loss"].append(avg_loss)
        run["train/accuracy"].append(accuracy)