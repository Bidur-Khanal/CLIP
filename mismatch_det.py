import os
import clip
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from finematch_dataset import FineMatchDataset
from utils import FastDataLoader, plot_per_sample_losses
import neptune
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pdb

class FineTuneCLIP(nn.Module):
    def __init__(self, clip_model, num_layers=3, hidden_size=128, feature_mode="mlp", fine_tune_clip=True):
        super(FineTuneCLIP, self).__init__()
        self.clip_model = clip_model
        self.feature_mode = feature_mode
        self.fine_tune_clip = fine_tune_clip

        # Freeze the CLIP backbone if fine_tune_clip is False
        if not self.fine_tune_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Determine input size based on feature mode
        self.projection_size = self.clip_model.text_projection.shape[1]
        if self.feature_mode == "concat":
            self.input_size = self.projection_size * 2  # For concatenation
        elif self.feature_mode == "channel_concat":
            self.input_size = self.projection_size  # For channel-wise concatenation (batch_size, 2, 512)
        else:
            self.input_size = self.projection_size  # For similarity

        # Define a simple MLP for classification
        layers = []
        input_size = self.input_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())  # Add a non-linear activation
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))  # Final layer for binary classification
        layers.append(nn.Sigmoid())  # Output activation
        self.mlp = nn.Sequential(*layers)

        # For handling channel-wise concatenation
        if self.feature_mode == "channel_concat":
            self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)  # 1x1 convolution

        # Ensure MLP matches the dtype of features
        self.mlp = self.mlp.to(torch.float16)

    def forward(self, images, texts):
        # Compute image and text features using CLIP
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(texts)

        # Normalize features
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

       

        # Combine features based on feature mode
        if self.feature_mode == "concat":
            features = torch.cat((image_features, text_features), dim=1)  # Shape: (batch_size, 1024)
        elif self.feature_mode == "channel_concat":
            features = torch.cat((image_features.unsqueeze(1), text_features.unsqueeze(1)), dim=1)  # Shape: (batch_size, 2, 512)
            features = self.conv(features).squeeze(1)  # Apply 1x1 convolution
        else:
            features = image_features * text_features  # Element-wise multiplication for similarity

        # Ensure MLP matches the dtype of features
        # self.mlp = self.mlp.to(features.dtype)
        # features = features.to(self.mlp[0].weight.dtype)

        out = self.mlp(features)

        return out




def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script with configurable parameters.")
    
    # General training parameters
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer to use (e.g., Adam, SGD)")
    parser.add_argument("--loss", type=str, default="BCELoss", help="Loss function to use")
    parser.add_argument("--fine_tune_clip", action="store_true", help="Fine-tune the CLIP model")
    

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
        "fine_tune_clip": args.fine_tune_clip,
    }

    # Example: Assign params to your tracking tool
    run["parameters"] = params
    run["all files"].upload_files("*.py")

    print(f"Parameters: {params}")


    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(pretrained_model, device)

    # Instantiate the fine-tuning model
    fine_tune_model = FineTuneCLIP(
        clip_model, num_layers=num_layers, hidden_size=hidden_size,
        feature_mode=args.feature_mode, fine_tune_clip=args.fine_tune_clip
    ).to(device)

    # Define optimizer and loss function
    # optimizer = torch.optim.Adam(fine_tune_model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(fine_tune_model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction="none")

    # Load datasets
    train_dataset = FineMatchDataset(train_jsonl, image_folder, transform=preprocess, include_list=include_list)
    val_dataset = FineMatchDataset(val_jsonl, image_folder, transform=preprocess, include_list=include_list)
    test_dataset = FineMatchDataset(test_jsonl, image_folder, transform=preprocess, include_list=include_list)

    train_loader = FastDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = FastDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = FastDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


    # set some variables
    best_val_accuracy = 0.0
    
    
    for epoch in range(num_epochs):

        ################### Training ###################
        fine_tune_model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        confidence_scores = []  # List to store confidence scores
        all_targets = []
        all_predictions = []
        all_outputs = []
        train_per_sample_losses = []  # Store for debugging purposes

        for images, queries, labels, labels_type,targets in train_loader:
            # print (labels)
            images, targets = images.to(device), targets.to(device, dtype=torch.float16)
            labels = clip.tokenize(labels).to(device)
            # print (labels)
            # pdb.set_trace()

            # Forward pass through the fine-tuning model
            outputs = fine_tune_model(images, labels).squeeze(1)
            # pdb.set_trace()
            per_sample_loss = criterion(outputs, targets)  # Compute per-sample loss
            loss = per_sample_loss.mean()  # Average loss for backpropagation

            # pdb.set_trace()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

            # Accumulate metrics
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct_train += (predictions == targets).sum().item()
            total_train += targets.size(0)

            # Store per-sample losses; used for debugging
            train_per_sample_losses.extend(per_sample_loss.detach().cpu().numpy())

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
        print(f"Epoch {epoch + 1}/{num_epochs}")

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
        fine_tune_model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        confidence_scores = []  # List to store confidence scores
        all_targets = []
        all_predictions = []
        all_outputs = []
        val_per_sample_losses = []  # Store for debugging purposes

        with torch.no_grad():
            for images, queries, labels, labels_type,targets in val_loader:
                images, targets = images.to(device), targets.to(device, dtype=torch.float16)
                labels = clip.tokenize(labels).to(device)

                # Forward pass through the fine-tuning model
                outputs = fine_tune_model(images, labels).squeeze(1)
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

        # Plot and log validation per-sample losses
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

            torch.save(fine_tune_model.state_dict(), best_model_path)
            print(f"    Best model saved at: {best_model_path} with validation accuracy {val_accuracy:.2f}%")



        ###################### Test #######################
        fine_tune_model.eval()
        test_loss, correct_test, total_test = 0.0, 0, 0
        confidence_scores = []  # List to store confidence scores
        all_targets = []
        all_predictions = []
        all_outputs = []
        test_per_sample_losses = []  # Store for debugging purposes

        with torch.no_grad():
            for images, queries, labels, labels_type,targets in test_loader:
                images, targets = images.to(device), targets.to(device, dtype=torch.float16)
                labels = clip.tokenize(labels).to(device)

                # Forward pass through the fine-tuning model
                outputs = fine_tune_model(images, labels).squeeze(1)
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

        # Plot and log test per-sample losses
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


    run.stop()