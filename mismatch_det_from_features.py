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



    # Dataset name (e.g., for dynamic path creation)
    dataset_name = "FineMatch"

    # Create directory for saving checkpoints
    checkpoint_dir = f"checkpoints/{dataset_name}"
    os.makedirs(checkpoint_dir, exist_ok=True) 


    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print (device)
    model, preprocess = clip.load('ViT-L/14', device)


    # File paths
    train_jsonl = "data_labels/FineMatch_train.jsonl"
    val_jsonl = "data_labels/FineMatch_val.jsonl"
    test_jsonl = "data_labels/FineMatch_test.jsonl"
    image_folder = "D:/finematch/images/FineMatch"  # Replace with the path to your image folder


    # Datasets
    train_dataset = FineMatchDataset(train_jsonl, image_folder, transform=preprocess)
    val_dataset = FineMatchDataset(val_jsonl, image_folder, transform=preprocess)
    test_dataset = FineMatchDataset(test_jsonl, image_folder, transform=preprocess)


    # Function to precompute embeddings
    def precompute_embeddings(dataset):
        image_embeddings, text_embeddings, targets_list = [], [], []
        dataloader = FastDataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        with torch.no_grad():
            for images, queries, labels, labels_type, targets in dataloader:
                image_inputs = images.to(device)
                text_inputs = clip.tokenize(labels).to(device)

                # Compute and normalize embeddings
                image_features = model.encode_image(image_inputs)
                text_features = model.encode_text(text_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Store embeddings and targets
                image_embeddings.append(image_features.cpu())
                text_embeddings.append(text_features.cpu())
                targets_list.append(targets.cpu())

        # Concatenate all embeddings and targets
        image_embeddings = torch.cat(image_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)
        targets_list = torch.cat(targets_list, dim=0)

        return torch.utils.data.TensorDataset(image_embeddings, text_embeddings, targets_list)

    print("Precomputing embeddings for train, val, and test datasets...")
    train_loader = DataLoader(precompute_embeddings(train_dataset), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(precompute_embeddings(val_dataset), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(precompute_embeddings(test_dataset), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    print("Precomputed embeddings for all datasets.")


    # Instantiate the full model
    classification_head = ClassificationHead(model).to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = torch.optim.Adam(classification_head.parameters(), lr=1e-2) # Optimizer

  
    num_epochs = 100
    best_val_accuracy = 0.0
    best_model_path = "checkpoints/best_model.pth"

    params = {"num_epochs": num_epochs, "lr": 1e-2, "batch_size": 128,
            "num_layers": 3, "hidden_size": 128, "pretrained_model": "ViT-L/14", 
            "dataset": "FineMatch", "optimizer": "Adam", "loss": "BCELoss"}
    run["parameters"] = params


    for epoch in range(num_epochs):

        ################### Training ###################
        classification_head.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for img_emb, txt_emb, targets in train_loader:
            img_emb, txt_emb, targets = img_emb.to(device), txt_emb.to(device), targets.to(device, dtype=torch.float)

            # Compute similarity
            similarity = img_emb * txt_emb

            # Forward pass
            outputs = classification_head(similarity).squeeze(1)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct_train += (predictions == targets).sum().item()
            total_train += targets.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = (correct_train / total_train) * 100


        ################# Validation  ####################
        classification_head.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for img_emb, txt_emb, targets in val_loader:
                img_emb, txt_emb, targets = img_emb.to(device), txt_emb.to(device), targets.to(device, dtype=torch.float)

                # Compute similarity
                similarity = img_emb * txt_emb

                # Forward pass
                outputs = classification_head(similarity).squeeze(1)
                loss = criterion(outputs, targets)

                # Accumulate metrics
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct_val += (predictions == targets).sum().item()
                total_val += targets.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_val / total_val) * 100

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(classification_head.state_dict(), best_model_path)
            print(f"Best model saved at: {best_model_path} with validation accuracy {val_accuracy:.2f}%")

        # Log metrics for train and val
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        run["train/loss"].append(avg_train_loss)
        run["train/accuracy"].append(train_accuracy)
        run["val/loss"].append(avg_val_loss)
        run["val/accuracy"].append(val_accuracy)


        ###################### Test  #######################
        classification_head.eval()
        test_loss, correct_test, total_test = 0.0, 0, 0
        with torch.no_grad():
            for img_emb, txt_emb, targets in test_loader:
                img_emb, txt_emb, targets = img_emb.to(device), txt_emb.to(device), targets.to(device, dtype=torch.float)

                # Compute similarity
                similarity = img_emb * txt_emb

                # Forward pass
                outputs = classification_head(similarity).squeeze(1)
                loss = criterion(outputs, targets)

                # Accumulate metrics
                test_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct_test += (predictions == targets).sum().item()
                total_test += targets.size(0)

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = (correct_test / total_test) * 100

        # Log test metrics
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        run["test/loss"] = avg_test_loss
        run["test/accuracy"] = test_accuracy

    run.stop()