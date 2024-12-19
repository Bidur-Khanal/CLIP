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
import argparse

class ClassificationHead(nn.Module):
    def __init__(self, clipmodel, num_layers=3, hidden_size=128):
        super(ClassificationHead, self).__init__()
        self.clipmodel = clipmodel
        self.projection_size = self.clipmodel.text_projection.shape[1]
       
        # Define a simple MLP
        self.layers = []
        self.num_layers = num_layers
        self.hidden_size = hidden_size
       
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
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script with configurable parameters.")
    
    # General training parameters
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer to use (e.g., Adam, SGD)")
    parser.add_argument("--loss", type=str, default="BCELoss", help="Loss function to use")
    parser.add_argument("--classification_type", type=str, default="zero-shot",choices= ["zero-shot", "mlp"]
                         ,help="Type of classification (e.g., zero-shot, mlp)")


    # Model parameters
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the model")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size for the model")
    parser.add_argument("--pretrained_model", type=str, default="ViT-L/14", help="Pretrained model to use")
    
    

    # File paths
    parser.add_argument("--train_jsonl", type=str, default="data_labels/FineMatch_train.jsonl",
                        help="Path to the training dataset JSONL file")
    parser.add_argument("--val_jsonl", type=str, default="data_labels/FineMatch_val.jsonl",
                        help="Path to the validation dataset JSONL file")
    parser.add_argument("--test_jsonl", type=str, default="data_labels/FineMatch_test.jsonl",
                        help="Path to the test dataset JSONL file")
    parser.add_argument("--image_folder", type=str, default="D:/finematch/images/FineMatch",
                        help="Path to the folder containing images")
    
    # Additional parameters
    parser.add_argument("--include_list", nargs="+", default=["Attributes", "Numbers", "Entities", "Relations"],
                        help="List of attributes to include in training")
    parser.add_argument("--dataset", type=str, default="FineMatch", help="Dataset name")

    return parser.parse_args()

    

if __name__ == "__main__":

    # Fetch the API token from the environment variable
    api_token = os.environ.get("NEPTUNE_API_TOKEN")
    
    if not api_token:
        raise ValueError("Neptune API token is not set. Please set the NEPTUNE_API_TOKEN environment variable.")
    
    # Initialize the Neptune run
    run = neptune.init_run(
    project="bidur/noisy-correspondence", api_token = api_token)
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Extract variables from arguments
    num_epochs = args.num_epochs
    include_list = args.include_list
    dataset_name = args.dataset
    batch_size = args.batch_size
    hiden_size = args.hidden_size
    num_layers = args.num_layers
    lr = args.lr
    pretrained_model = args.pretrained_model
    train_jsonl = args.train_jsonl
    val_jsonl = args.val_jsonl
    test_jsonl = args.test_jsonl
    image_folder = args.image_folder

    # Create directory for saving checkpoints
    checkpoint_dir = f"checkpoints/{dataset_name}"
    os.makedirs(checkpoint_dir, exist_ok=True) 


    # Create params dictionary
    params = {
        "num_epochs": num_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "pretrained_model": args.pretrained_model,
        "dataset": args.dataset,
        "optimizer": args.optimizer,
        "loss": args.loss,
        "include_list": str(args.include_list),
    }

    # Example: Assign params to your tracking tool
    run["parameters"] = params
    print(f"Parameters: {params}")


    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(pretrained_model, device)


    # Datasets
    train_dataset = FineMatchDataset(train_jsonl, image_folder, transform=preprocess, include_list=include_list)
    val_dataset = FineMatchDataset(val_jsonl, image_folder, transform=preprocess, include_list=include_list)
    test_dataset = FineMatchDataset(test_jsonl, image_folder, transform=preprocess, include_list=include_list)


    # Function to precompute embeddings
    def precompute_embeddings(dataset):
        image_embeddings, text_embeddings, targets_list = [], [], []
        dataloader = FastDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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
    train_loader = DataLoader(precompute_embeddings(train_dataset), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(precompute_embeddings(val_dataset), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(precompute_embeddings(test_dataset), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("Precomputed embeddings for all datasets.")


    # Instantiate the full model
    classification_head = ClassificationHead(model,num_layers,hiden_size).to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = torch.optim.Adam(classification_head.parameters(), lr=lr) # Optimizer

    # set some variables
    best_val_accuracy = 0.0
    
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

            # Determine best_model_path based on include_list
            if include_list == ["Attributes"]:
                best_model_path = os.path.join(checkpoint_dir, "Attributes/best_model.pth")
            elif include_list == ["Numbers"]:
                best_model_path = os.path.join(checkpoint_dir, "Numbers/best_model.pth")
            elif include_list == ["Relations"]:
                best_model_path = os.path.join(checkpoint_dir, "Relations/best_model.pth")
            elif include_list == ["Entities"]:
                best_model_path = os.path.join(checkpoint_dir, "Entities/best_model.pth")
            elif set(include_list) == {"Attributes", "Numbers", "Entities", "Relations"}:
                best_model_path = os.path.join(checkpoint_dir, "All/best_model.pth")
            else:
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

            # Ensure the directory for best_model_path exists
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

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
        run["test/loss"].append(avg_test_loss)
        run["test/accuracy"].append(test_accuracy)

    run.stop()