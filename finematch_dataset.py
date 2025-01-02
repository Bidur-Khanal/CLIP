import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import defaultdict
from utils import FastDataLoader
import clip

class FineMatchDataset(Dataset):
    def __init__(self, jsonl_file, image_folder, transform=None, tokenizer=None, include_list = ["Attributes","Numbers","Entities","Relations"]):
        """
        Custom Dataset for loading images, queries, labels, and targets.

        Parameters:
        - jsonl_file (str): Path to the processed JSONL file.
        - image_folder (str): Path to the folder containing images.
        - transform (callable, optional): Transformation to apply to the images.
        - tokenizer (callable, optional): Tokenizer for processing the queries.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = tokenizer
        self.include_list = include_list 
        print (self.include_list)
        
        # Load data from the JSONL file
        with open(jsonl_file, "r") as f:
            self.data = [json.loads(line) for line in f]

        # Filter out entries with label types not in include_list
        self.data = [entry for entry in self.data if entry["label_type"] in self.include_list]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the data for the given index
        entry = self.data[idx]
        image_id = entry["image_id"]
        query = entry["query"]
        label = entry["labels"]
        target = entry["target"]
        label_type = entry["label_type"]

        # Check if the image exists in the folder
        image_path = os.path.join(self.image_folder, image_id)
        if not os.path.exists(image_path):
            # Modify the image_id by prepending "VizWiz_train_"
            image_id = f"VizWiz_train_{image_id}"
            image_path = os.path.join(self.image_folder, image_id)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_id}")


        # Load the image
        image_path = os.path.join(self.image_folder, image_id)
        image = Image.open(image_path).convert("RGB")

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        # Tokenize the query if a tokenizer is provided
        if self.tokenizer:
            query = self.tokenizer(query)

        # Return the image, query, label, and target
        return image, query, label, label_type, target
    
class FineMatchZeroShotDataset(Dataset):
    def __init__(self, jsonl_file, image_folder, transform=None, tokenizer=None, include_list=["Attributes", "Numbers", "Entities", "Relations"]):
        """
        Custom Dataset for loading images, queries, labels, and targets.

        Parameters:
        - jsonl_file (str): Path to the processed JSONL file.
        - image_folder (str): Path to the folder containing images.
        - transform (callable, optional): Transformation to apply to the images.
        - tokenizer (callable, optional): Tokenizer for processing the queries.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = tokenizer
        self.include_list = include_list
        print (self.include_list)

        # Load data from the JSONL file
        with open(jsonl_file, "r") as f:
            data = [json.loads(line) for line in f]

        # Filter out entries with label types not in include_list
        data = [entry for entry in data if entry["label_type"] in self.include_list]

        # Group every two consecutive lines
        self.data = []

        for i in range(0, len(data), 2):
            entry1 = data[i]
            entry2 = data[i + 1]
            
            # Create a new grouped entry
            grouped_entry = {
                "id": entry1["id"],
                "image_id": entry1["image_id"],
                "query": entry1["query"],
                "labels": [entry1["labels"], entry2["labels"]],
                "targets": [entry1["target"], entry2["target"]],
                "label_type": entry1["label_type"]
            }
    
            self.data.append(grouped_entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the data for the given index
        entry = self.data[idx]
        image_id = entry["image_id"]
        query = entry["query"]
        label_type = entry["label_type"]
        labels = entry["labels"]
        targets = entry["targets"]

        # Check if the image exists in the folder
        image_path = os.path.join(self.image_folder, image_id)
        if not os.path.exists(image_path):
            # Modify the image_id by prepending "VizWiz_train_"
            image_id = f"VizWiz_train_{image_id}"
            image_path = os.path.join(self.image_folder, image_id)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_id}")

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        # Tokenize the query if a tokenizer is provided
        if self.tokenizer:
            query = self.tokenizer(query)

        # Return the image, query, combined labels, label type, and targets
        return image, query, labels, label_type, targets
    

# Example usage
if __name__ == "__main__":
    # File paths
    jsonl_file = "data_labels/FineMatch_train.jsonl"  # Replace with the path to your processed JSONL file
    image_folder = "D:/finematch/images/FineMatch"  # Replace with the path to your image folder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    # Define image transformations
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # Load the tokenizer for text queries
    # tokenizer = clip.tokenize

    # Create the dataset
    dataset = FineMatchDataset(jsonl_file, image_folder, transform=preprocess)

    # Create the dataloader
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    dataloader = FastDataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # Iterate through the dataloader
    for images, queries, labels, labels_type,targets in dataloader:
        print(f"Images shape: {images.shape}")  # Image tensors
        print(f"Queries: {queries}")          # Tokenized queries
        print(f"Labels: {labels}")            # Labels
        print(f"Targets: {targets}")          # Targets (0 or 1)
        print(f"Label Type: {labels_type}")    # Label Type

