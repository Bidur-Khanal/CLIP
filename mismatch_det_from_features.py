import os
import clip
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from finematch_dataset import FineMatchDataset, FineMatchZeroShotDataset
from utils import FastDataLoader, plot_per_sample_losses
import neptune
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class ClassificationHead(nn.Module):
    def __init__(self, clipmodel, num_layers=3, hidden_size=128, feature_mode="similarity"):
        super(ClassificationHead, self).__init__()
        self.clipmodel = clipmodel
        self.feature_mode = feature_mode

        # Determine input size based on feature mode
        self.projection_size = self.clipmodel.text_projection.shape[1]
        if self.feature_mode == "concat":
            self.input_size = self.projection_size * 2  # For concatenation
        elif self.feature_mode == "channel_concat":
            self.input_size = self.projection_size  # For channel-wise concatenation (batch_size, 2, 512)
        else:
            self.input_size = self.projection_size  # For similarity

        # Define a simple MLP
        self.layers = []
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        input_size = self.input_size
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(input_size, self.hidden_size))
            self.layers.append(nn.ReLU())  # Add a non-linear activation
            input_size = self.hidden_size

        self.layers.append(nn.Linear(input_size, 1))  # Final layer for binary classification
        self.layers.append(nn.Sigmoid())  # Output activation

        self.mlp = nn.Sequential(*self.layers)

        # For handling channel-wise concatenation
        if self.feature_mode == "channel_concat":
            self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)  # 1x1 convolution

    def forward(self, features):
        # Ensure the input is in the correct dtype
        features = features.type(self.mlp[0].weight.dtype)  # Match the first layer's weight dtype

        # Apply 1x1 convolution for channel-wise concatenation
        if self.feature_mode == "channel_concat":
            features = self.conv(features).squeeze(1)  # Shape: (batch_size, 512)

        return self.mlp(features)
    

# Function to precompute embeddings for mlp classification
def precompute_embeddings(dataset, batch_size, device, model):
    image_embeddings, text_embeddings, targets_list = [], [], []
    dataloader = FastDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for images, queries, labels, labels_type, targets in dataloader:

            image_inputs = images.to(device)
            text_inputs = clip.tokenize(labels).to(device)

            # Normalize image embeddings
            image_features = model.encode_image(image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Append image embeddings
            image_embeddings.append(image_features.cpu())
            
            # Normalize text embeddings
            text_features = model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Append text embeddings and targets
            text_embeddings.append(text_features.cpu())
            targets_list.append(targets)

    # Concatenate all embeddings and targets
    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    targets_list = torch.cat(targets_list, dim=0)

    return torch.utils.data.TensorDataset(image_embeddings, text_embeddings, targets_list)


# Function to precompute embeddings for zero-shot classification
def precompute_embeddings_zero_shot(dataset, batch_size, device, model):
    image_embeddings, text_embeddings1, text_embeddings2 = [], [], []
    targets_list1, targets_list2 = [], []
    dataloader = FastDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for images, queries, labels, labels_type, targets in dataloader:

            image_inputs = images.to(device)

            # Normalize image embeddings and append to list
            image_features = model.encode_image(image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_embeddings.append(image_features.cpu())

            # Tokenize and compute text embeddings for labels1
            text_inputs1 = clip.tokenize(labels[0]).to(device)
            text_features1 = model.encode_text(text_inputs1)
            text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
            text_embeddings1.append(text_features1.cpu())

            # Tokenize and compute text embeddings for labels2
            text_inputs2 = clip.tokenize(labels[1]).to(device)
            text_features2 = model.encode_text(text_inputs2)
            text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
            text_embeddings2.append(text_features2.cpu())

            # Store targets
            targets_list1.append(targets[0])
            targets_list2.append(targets[1])

    # Concatenate all embeddings and targets
    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings1 = torch.cat(text_embeddings1, dim=0)
    text_embeddings2 = torch.cat(text_embeddings2, dim=0)
    targets_list1 = torch.cat(targets_list1, dim=0)
    targets_list2 = torch.cat(targets_list2, dim=0)

    return torch.utils.data.TensorDataset(image_embeddings, text_embeddings1, text_embeddings2, targets_list1, targets_list2)

    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script with configurable parameters.")
    
    # General training parameters
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer to use (e.g., Adam, SGD)")
    parser.add_argument("--loss", type=str, default="BCELoss", help="Loss function to use")
    parser.add_argument("--classification_type", type=str, default="mlp",choices= ["zero-shot", "mlp"]
                         ,help="Type of classification (e.g., zero-shot, mlp)")


    # Model parameters
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the model")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size for the model")
    parser.add_argument("--pretrained_model", type=str, default="ViT-L/14", help="Pretrained model to use")

    # Feature mode
    parser.add_argument("--feature_mode", type=str, default="concat", choices=["similarity", "concat", "channel_concat"],
                        help="Feature mode: 'similarity' for element-wise multiplication, 'concat' for concatenation,\
                              'channel_concat' for channel-wise concatenation")
    
    

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
                        help="List of attributes to include in training: Example --include_list Attributes Numbers Entities Relations \
                            or --include_list Attributes Numbers")
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
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    lr = args.lr
    pretrained_model = args.pretrained_model
    train_jsonl = args.train_jsonl
    val_jsonl = args.val_jsonl
    test_jsonl = args.test_jsonl
    image_folder = args.image_folder
    classification_type = args.classification_type

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
        "feature_mode": args.feature_mode,
        "classification_type": args.classification_type,
    }

    # Example: Assign params to your tracking tool
    run["parameters"] = params
    run["all files"].upload_files("*.py")

    print(f"Parameters: {params}")


    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(pretrained_model, device)


    if classification_type == "mlp":

        print("Precomputing embeddings for train, val, and test datasets...")
        train_dataset = FineMatchDataset(train_jsonl, image_folder, transform=preprocess, include_list=include_list)
        val_dataset = FineMatchDataset(val_jsonl, image_folder, transform=preprocess, include_list=include_list)
        test_dataset = FineMatchDataset(test_jsonl, image_folder, transform=preprocess, include_list=include_list)


        train_loader = DataLoader(precompute_embeddings(train_dataset,batch_size=batch_size,device=device,model=model),\
                                batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(precompute_embeddings(val_dataset,batch_size=batch_size,device=device,model=model),\
                                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(precompute_embeddings(test_dataset, batch_size=batch_size,device=device,model=model),\
                                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        print("Precomputed embeddings for all datasets.")
        

        # Instantiate the model
        classification_head = ClassificationHead(model, num_layers,hidden_size,args.feature_mode).to(device)

        # Loss function and optimizer
        criterion = nn.BCELoss(reduction='none')  # Use reduction='none' to compute per-sample loss
        optimizer = torch.optim.Adam(classification_head.parameters(), lr=lr) # Optimizer

        # set some variables
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):

            ################### Training ###################
            classification_head.train()
            train_loss, correct_train, total_train = 0.0, 0, 0
            confidence_scores = []  # List to store confidence scores
            all_targets = []
            all_predictions = []
            all_outputs = []
            train_per_sample_losses = [] # Store for debugging purposes

            for img_emb, txt_emb, targets in train_loader:
                img_emb, txt_emb, targets = img_emb.to(device), txt_emb.to(device), targets.to(device, dtype=torch.float)

                # Compute features based on feature mode
                if args.feature_mode == "concat":
                    features = torch.cat((img_emb, txt_emb), dim=1)  # Shape: (batch_size, 1024)
                elif args.feature_mode == "channel_concat":
                    features = torch.cat((img_emb.unsqueeze(1), txt_emb.unsqueeze(1)), dim=1)  # Shape: (batch_size, 2, 512)
                else:
                    features = img_emb * txt_emb  # Element-wise multiplication for similarity

                # Forward pass
                outputs = classification_head(features).squeeze(1)
                per_sample_loss = criterion(outputs, targets)  # Compute per-sample loss
                loss = per_sample_loss.mean()  # Average loss for backpropagation

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store per-sample losses; used for debugging
                train_per_sample_losses.extend(per_sample_loss.detach().cpu().numpy())


                # Accumulate metrics
                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct_train += (predictions == targets).sum().item()
                total_train += targets.size(0)

                # Collect predictions and targets for metrics
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

                # Calculate confidence and store it
                confidences = torch.max(outputs.detach(), 1 - outputs.detach())  # Confidence scores for each prediction
                confidence_scores.extend(confidences.cpu().numpy())

            # Compute additional metrics
            train_precision = precision_score(all_targets, all_predictions)
            train_recall = recall_score(all_targets, all_predictions)
            train_f1 = f1_score(all_targets, all_predictions)
            train_roc_auc = roc_auc_score(all_targets, all_outputs)

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = (correct_train / total_train) * 100
            avg_train_confidence = sum(confidence_scores) / len(confidence_scores)

            # Plot and log train per-sample losses
            fig = plot_per_sample_losses(train_per_sample_losses)
            run["train/per_sample_losses_plot"].log(neptune.types.File.as_image(fig))
            plt.close(fig)


            # Log metrics for train and val
            print(f"Epoch {epoch+1}/{num_epochs}")

            # Log metrics for training
            print(f"    Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Train Confidence: {avg_train_confidence:.4f}")
            print(f"    Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1-Score: {train_f1:.4f}, Train ROC-AUC: {train_roc_auc:.4f}")

            run["train/loss"].append(avg_train_loss)
            run["train/accuracy"].append(train_accuracy)
            run["train/confidence"].append(avg_train_confidence)
            run["train/precision"].append(train_precision)
            run["train/recall"].append(train_recall)
            run["train/f1"].append(train_f1)
            run["train/roc_auc"].append(train_roc_auc)



            ################# Validation ####################
            classification_head.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0
            confidence_scores = []  # List to store confidence scores
            all_targets = []
            all_predictions = []
            all_outputs = []
            val_per_sample_losses = []  # Store for debugging purposes

            with torch.no_grad():
                for img_emb, txt_emb, targets in val_loader:
                    img_emb, txt_emb, targets = img_emb.to(device), txt_emb.to(device), targets.to(device, dtype=torch.float)

                    # Compute features based on feature mode
                    if args.feature_mode == "concat":
                        features = torch.cat((img_emb, txt_emb), dim=1)
                    elif args.feature_mode == "channel_concat":
                        features = torch.cat((img_emb.unsqueeze(1), txt_emb.unsqueeze(1)), dim=1)
                    else:
                        features = img_emb * txt_emb

                    # Forward pass
                    outputs = classification_head(features).squeeze(1)
                    per_sample_loss = criterion(outputs, targets)  # Compute per-sample loss
                    loss = per_sample_loss.mean()  # Average loss for metrics

                    # Store per-sample losses; used for debugging
                    val_per_sample_losses.extend(per_sample_loss.detach().cpu().numpy())

                    # Accumulate metrics
                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    correct_val += (predictions == targets).sum().item()
                    total_val += targets.size(0)

                    # Collect predictions and targets for metrics
                    all_targets.extend(targets.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
                    all_outputs.extend(outputs.cpu().numpy())

                    # Calculate confidence and store it
                    confidences = torch.max(outputs, 1 - outputs)  # Confidence scores for each prediction
                    confidence_scores.extend(confidences.cpu().numpy())

            # Compute additional metrics
            val_precision = precision_score(all_targets, all_predictions)
            val_recall = recall_score(all_targets, all_predictions)
            val_f1 = f1_score(all_targets, all_predictions)
            val_roc_auc = roc_auc_score(all_targets, all_outputs)

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = (correct_val / total_val) * 100
            avg_val_confidence = sum(confidence_scores) / len(confidence_scores)


            # Plot and log train per-sample losses
            fig = plot_per_sample_losses(val_per_sample_losses)
            run["val/per_sample_losses_plot"].log(neptune.types.File.as_image(fig))
            plt.close(fig)

            # Log metrics for validation
            print(f"    Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Val Confidence: {avg_val_confidence:.4f}")
            print(f"    Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1-Score: {val_f1:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")

            run["val/loss"].append(avg_val_loss)
            run["val/accuracy"].append(val_accuracy)
            run["val/confidence"].append(avg_val_confidence)
            run["val/precision"].append(val_precision)
            run["val/recall"].append(val_recall)
            run["val/f1"].append(val_f1)
            run["val/roc_auc"].append(val_roc_auc)


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
                print(f"    Best model saved at: {best_model_path} with validation accuracy {val_accuracy:.2f}%")



            ###################### Test #######################
            classification_head.eval()
            test_loss, correct_test, total_test = 0.0, 0, 0
            confidence_scores = []  # List to store confidence scores
            all_targets = []
            all_predictions = []
            all_outputs = []
            test_per_sample_losses = []  # Store for debugging purposes

            with torch.no_grad():
                for img_emb, txt_emb, targets in test_loader:
                    img_emb, txt_emb, targets = img_emb.to(device), txt_emb.to(device), targets.to(device, dtype=torch.float)

                    # Compute features based on feature mode
                    if args.feature_mode == "concat":
                        features = torch.cat((img_emb, txt_emb), dim=1)
                    elif args.feature_mode == "channel_concat":
                        features = torch.cat((img_emb.unsqueeze(1), txt_emb.unsqueeze(1)), dim=1)
                    else:
                        features = img_emb * txt_emb

                    # Forward pass
                    outputs = classification_head(features).squeeze(1)
                    per_sample_loss = criterion(outputs, targets)  # Compute per-sample loss
                    loss = per_sample_loss.mean()  # Average loss for metrics

                    # Store per-sample losses; used for debugging
                    test_per_sample_losses.extend(per_sample_loss.detach().cpu().numpy())

                    # Accumulate metrics
                    test_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    correct_test += (predictions == targets).sum().item()
                    total_test += targets.size(0)

                    # Collect predictions and targets for metrics
                    all_targets.extend(targets.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
                    all_outputs.extend(outputs.cpu().numpy())

                    # Calculate confidence and store it
                    confidences = torch.max(outputs, 1 - outputs)  # Confidence scores for each prediction
                    confidence_scores.extend(confidences.cpu().numpy())

            # Compute additional metrics
            test_precision = precision_score(all_targets, all_predictions)
            test_recall = recall_score(all_targets, all_predictions)
            test_f1 = f1_score(all_targets, all_predictions)
            test_roc_auc = roc_auc_score(all_targets, all_outputs)

            avg_test_loss = test_loss / len(test_loader)
            test_accuracy = (correct_test / total_test) * 100
            avg_test_confidence = sum(confidence_scores) / len(confidence_scores)

            # Plot and log train per-sample losses
            fig = plot_per_sample_losses(test_per_sample_losses)
            run["test/per_sample_losses_plot"].log(neptune.types.File.as_image(fig))
            plt.close(fig)

            # Log test metrics
            print(f"    Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test Confidence: {avg_test_confidence:.4f}")
            print(f"    Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1-Score: {test_f1:.4f}, Test ROC-AUC: {test_roc_auc:.4f}")

            run["test/loss"].append(avg_test_loss)
            run["test/accuracy"].append(test_accuracy)
            run["test/confidence"].append(avg_test_confidence)
            run["test/precision"].append(test_precision)
            run["test/recall"].append(test_recall)
            run["test/f1"].append(test_f1)
            run["test/roc_auc"].append(test_roc_auc)


    elif classification_type == "zero-shot":


        print("Precomputing embeddings for train, val, and test datasets for zero-shot clasification...")

        train_dataset = FineMatchZeroShotDataset(train_jsonl, image_folder, transform=preprocess, include_list=include_list)
        val_dataset = FineMatchZeroShotDataset(val_jsonl, image_folder, transform=preprocess, include_list=include_list)
        test_dataset = FineMatchZeroShotDataset(test_jsonl, image_folder, transform=preprocess, include_list=include_list)

        train_loader = DataLoader(precompute_embeddings_zero_shot(train_dataset,batch_size=batch_size,device=device,model=model),\
                                batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(precompute_embeddings_zero_shot(val_dataset,batch_size=batch_size,device=device,model=model),\
                                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(precompute_embeddings_zero_shot(test_dataset, batch_size=batch_size,device=device,model=model),\
                                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        print("Precomputed embeddings for all datasets.")

        ################### Train set ###################
        correct_train, total_train = 0, 0
        confidence_scores = []  # List to store confidence scores
        all_targets = []
        all_predictions = []
        all_outputs = []

        for img_emb, txt_emb1, txt_emb2, targets1, targets2 in train_loader:
            img_emb, txt_emb1, txt_emb2 = img_emb.to(device), txt_emb1.to(device), txt_emb2.to(device)
            targets1, targets2 = targets1.to(device, dtype=torch.float), targets2.to(device, dtype=torch.float)

            # Compute image-text similarity for first text
            similarity1 = torch.sum(img_emb * txt_emb1, dim =1)

            # Compute image-text similarity for second text
            similarity2 = torch.sum(img_emb * txt_emb2, dim =1)

            # Stack similarities and apply softmax to get probabilities
            similarities = torch.stack([similarity1, similarity2], dim=1)  # Shape: (batch_size, 2)
            probabilities = torch.softmax(similarities, dim=1)  # Shape: (batch_size, 2)

            # Extract probabilities for each similarity
            prob1 = probabilities[:, 0]  # Probability for similarity1
            prob2 = probabilities[:, 1]  # Probability for similarity2

            # Generate predictions: 0 if prob1 > prob2, else 1
            predictions = (prob2 > prob1).float()

            # Accumulate metrics
            correct_train += (predictions == targets2).sum().item()
            total_train += targets2.size(0)

            # Collect predictions and targets for metrics
            all_targets.extend(targets2.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    

            # Store confidence scores (max of prob1 and prob2)
            confidences = torch.max(prob1, prob2)
            confidence_scores.extend(confidences.cpu().numpy())

        # Compute additional metrics
        train_precision = precision_score(all_targets, all_predictions)
        train_recall = recall_score(all_targets, all_predictions)
        train_f1 = f1_score(all_targets, all_predictions)
        train_accuracy = (correct_train / total_train) * 100

        # Log metrics for training
        print(f"    Train Accuracy: {train_accuracy:.2f}%")
        print(f"    Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1-Score: {train_f1:.4f}")

        run["train/accuracy"].append(train_accuracy)
        run["train/precision"].append(train_precision)
        run["train/recall"].append(train_recall)
        run["train/f1"].append(train_f1)
        


        ################### Validation Set ###################
        correct_val, total_val = 0, 0
        confidence_scores = []  # List to store confidence scores
        all_targets = []
        all_predictions = []
        all_outputs = []

        with torch.no_grad():
            for img_emb, txt_emb1, txt_emb2, targets1, targets2 in val_loader:
                img_emb, txt_emb1, txt_emb2 = img_emb.to(device), txt_emb1.to(device), txt_emb2.to(device)
                targets1, targets2 = targets1.to(device, dtype=torch.float), targets2.to(device, dtype=torch.float)

                # Compute image-text similarity for first text
                similarity1 = torch.sum(img_emb * txt_emb1, dim =1)

                # Compute image-text similarity for second text
                similarity2 = torch.sum(img_emb * txt_emb2, dim =1)

                # Stack similarities and apply softmax to get probabilities
                similarities = torch.stack([similarity1, similarity2], dim=1)  # Shape: (batch_size, 2)
                probabilities = torch.softmax(similarities, dim=1)  # Shape: (batch_size, 2)

                # Extract probabilities for each similarity
                prob1 = probabilities[:, 0]  # Probability for similarity1
                prob2 = probabilities[:, 1]  # Probability for similarity2

                # Generate predictions: 0 if prob1 > prob2, else 1
                predictions = (prob2 > prob1).float()

                # Accumulate metrics
                correct_val += (predictions == targets2).sum().item()
                total_val += targets2.size(0)

                # Collect predictions and targets for metrics
                all_targets.extend(targets2.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

                # Store confidence scores (max of prob1 and prob2)
                confidences = torch.max(prob1, prob2)
                confidence_scores.extend(confidences.cpu().numpy())

        # Compute additional metrics
        val_precision = precision_score(all_targets, all_predictions)
        val_recall = recall_score(all_targets, all_predictions)
        val_f1 = f1_score(all_targets, all_predictions)
        val_accuracy = (correct_val / total_val) * 100

        # Log metrics for validation
        print(f"    Val Accuracy: {val_accuracy:.2f}%")
        print(f"    Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1-Score: {val_f1:.4f}")

        run["val/accuracy"].append(val_accuracy)
        run["val/precision"].append(val_precision)
        run["val/recall"].append(val_recall)
        run["val/f1"].append(val_f1)
       


        ################### Test Set ###################
        correct_test, total_test = 0, 0
        confidence_scores = []  # List to store confidence scores
        all_targets = []
        all_predictions = []
      
        with torch.no_grad():
            for img_emb, txt_emb1, txt_emb2, targets1, targets2 in test_loader:
                img_emb, txt_emb1, txt_emb2 = img_emb.to(device), txt_emb1.to(device), txt_emb2.to(device)
                targets1, targets2 = targets1.to(device, dtype=torch.float), targets2.to(device, dtype=torch.float)

                # Compute image-text similarity for first text
                similarity1 = torch.sum(img_emb * txt_emb1, dim =1)

                # Compute image-text similarity for second text
                similarity2 = torch.sum(img_emb * txt_emb2, dim =1)

                # Stack similarities and apply softmax to get probabilities
                similarities = torch.stack([similarity1, similarity2], dim=1)  # Shape: (batch_size, 2)
                probabilities = torch.softmax(similarities, dim=1)  # Shape: (batch_size, 2)

                # Extract probabilities for each similarity
                prob1 = probabilities[:, 0]  # Probability for similarity1
                prob2 = probabilities[:, 1]  # Probability for similarity2

                # Generate predictions: 0 if prob1 > prob2, else 1
                predictions = (prob2 > prob1).float()

                # Accumulate metrics
                correct_test += (predictions == targets2).sum().item()
                total_test += targets2.size(0)

                # Collect predictions and targets for metrics
                all_targets.extend(targets2.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

                # Store confidence scores (max of prob1 and prob2)
                confidences = torch.max(prob1, prob2)
                confidence_scores.extend(confidences.cpu().numpy())

        # Compute additional metrics
        test_precision = precision_score(all_targets, all_predictions)
        test_recall = recall_score(all_targets, all_predictions)
        test_f1 = f1_score(all_targets, all_predictions)
        test_accuracy = (correct_test / total_test) * 100

        # Log metrics for test
        print(f"    Test Accuracy: {test_accuracy:.2f}%")
        print(f"    Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1-Score: {test_f1:.4f}")

        run["test/accuracy"].append(test_accuracy)
        run["test/precision"].append(test_precision)
        run["test/recall"].append(test_recall)
        run["test/f1"].append(test_f1)
     

    run.stop()