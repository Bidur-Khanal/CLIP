import json

def process_jsonl(input_file, output_file):
    """
    Processes the input JSONL file and generates an output JSONL file
    with the specified format.

    Parameters:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to the output JSONL file.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)

            # Skip if "labels" is "None"
            if data.get("labels") == "None":
                continue
            
            # Iterate over each label group
            for label_group in data["labels"]:
                label_type, label_0, label_1 = label_group
                
                # Create two entries: one for target=0 and one for target=1
                for label, target in [(label_0, 0), (label_1, 1)]:
                    output_data = {
                        "id": data["id"],
                        "image_id": data["image_id"],
                        "query": data["query"],
                        "labels": label,
                        "target": target,
                        "label_type": label_type
                    }
                    # Write the processed entry to the output file
                    outfile.write(json.dumps(output_data) + '\n')

# Specify input and output files
input_file = "D:/finematch/FineMatch_test.jsonl"  # Replace with your input JSONL file path
output_file = "data_labels/FineMatch_test.jsonl"  # Replace with your desired output file path

# Process the JSONL file
process_jsonl(input_file, output_file)

print(f"Processed file saved to {output_file}")
