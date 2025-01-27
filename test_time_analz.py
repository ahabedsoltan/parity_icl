import torch
import matplotlib.pyplot as plt
from transformers import GPT2Config, GPT2LMHeadModel
import ipdb
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.OOD import create_dataset
from tqdm import tqdm
from itertools import combinations

import plotly.graph_objects as go
import wandb
import numpy as np

import wandb
from collections import defaultdict
import os
import re
import random
import argparse

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Description of your script.")

# Add arguments

parser.add_argument('--dim_k', type=int, default=3)
parser.add_argument('--dim', type=int, default=15)
parser.add_argument('--cl', type=int, default=40)


# Parse arguments
args = parser.parse_args()


wandb_init = {}
wandb_init["project_name"] = "COT-ICL_test_time"
wandb_init["mode"] = 'disabled'
wandb_init["key"] = "1840d187a4e8cfe03c2d82a3ff679f732d94b304"
wandb_init["org"] = "belkinlab"

os.environ["WANDB_API_KEY"] = wandb_init['key']
os.environ["WANDB_MODE"] = wandb_init['mode']  # online or offline

run = wandb.init(project=wandb_init['project_name'], entity=wandb_init['org'])



from collections import defaultdict

def filter_last_k_torch(A, mask, k):
    result = []
    for row, row_mask in zip(A, mask):
        # Get indices where mask is True
        true_indices = torch.nonzero(row_mask).squeeze(1)
        # Select the last k True elements
        last_k_indices = true_indices[-k:]
        # Extract corresponding elements from the row
        result.append(row[last_k_indices])
    return torch.vstack(result)

def evaluate_model(model, val_loader, device, k=3):
    """
    Evaluate the model on the validation set with per-key and per-coordinate accuracy.

    Args:
        model: The PyTorch model to evaluate.
        val_loader: DataLoader for the validation set.
        device: The device ('cpu' or 'cuda') to perform evaluation on.
        k: Number of last coordinates to consider for accuracy.

    Returns:
        result: Dictionary containing average validation loss and accuracies:
                {
                    "accuracy_all": Accuracy when all last k+1 coordinates match,
                    "accuracy_0": Accuracy for the last coordinate,
                    ...,
                    "accuracy_per_key": Dictionary with accuracies for each key,
                    "accuracy_per_key_per_coord": Dictionary with accuracies for each key and each coordinate.
                }
    """
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct_all = 0
    correct_coords = [0] * (k + 1)
    total = 0

    # Dictionary to store per-key statistics
    key_stats = {}
    key_coord_stats = {}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation Progress", unit="batch"):

            inputs, targets, keys = batch['input_ids'], batch['labels'], batch['keys']
            inputs, targets = inputs.to(device), targets.to(device)

            shifted_targets = targets[:, 1:]
            inputs = inputs[:, :-1]

            # Forward pass
            outputs = model(input_ids=inputs)
            logits = outputs.logits
            logits = torch.argmax(logits, dim=-1)

            valid_mask = shifted_targets != -100
            filtered_logits = filter_last_k_torch(logits, valid_mask, k + 1)
            filtered_targets = filter_last_k_torch(shifted_targets, valid_mask, k + 1)

            # Compare each coordinate and the full match
            matches_all = torch.all(filtered_logits == filtered_targets, dim=1)
            correct_all += matches_all.sum().item()

            for i in range(k + 1):
                matches_coord = (filtered_logits[:, i] == filtered_targets[:, i])
                correct_coords[i] += matches_coord.sum().item()

            total += inputs.size(0)

            # Per-key and per-coordinate accuracy calculation
            for idx, key in enumerate(keys):
                key = key.item() if hasattr(key, 'item') else key
                if key not in key_stats:
                    key_stats[key] = {'correct': 0, 'total': 0}
                    key_coord_stats[key] = [0] * (k + 1)

                key_stats[key]['total'] += 1
                if matches_all[idx].item():
                    key_stats[key]['correct'] += 1

                for i in range(k + 1):
                    if filtered_logits[idx, i] == filtered_targets[idx, i]:
                        key_coord_stats[key][i] += 1

    # Calculate metrics
    # avg_loss = val_loss / total
    accuracy_all = correct_all / total
    accuracy_coords = [correct / total for correct in correct_coords]

    # Prepare results
    result = {"accuracy_all": accuracy_all}
    for i in range(k + 1):
        result[f"accuracy_{i}"] = accuracy_coords[i]

    # Per-key accuracies
    result['accuracy_per_key'] = {
        key: stats['correct'] / stats['total'] for key, stats in key_stats.items()
    }

    # Per-key per-coordinate accuracies
    result['accuracy_per_key_per_coord'] = {
        key: [key_coord_stats[key][i] / key_stats[key]['total'] for i in range(k + 1)]
        for key in key_stats
    }


    return result



config = GPT2Config(
    vocab_size=2,  # Binary tokens: 0 and 1
    n_embd=192,  # Embedding dimension (default GPT-2)
    n_layer=3,  # Number of transformer layers
    n_head=1,  # Number of attention heads
    n_positions=2048
)



model = GPT2LMHeadModel(config)

# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint, strict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path = "/u/abedsol1/research/parity_icl/jsons_files/d15_k3.json"
# Load the JSON file
with open(file_path, "r") as f:
    data = json.load(f)

k = args.dim_k#data.get("dim_k", {}).get("value", [])
d = args.dim#data.get("dim_ambient", {}).get("value", [])
remaining_sequences = data.get("n_test_seqs", {}).get("value", [])
remaining_sequences = random.sample(remaining_sequences, min(500,len(remaining_sequences))  )
# context_length = 40

train_set = np.array(data.get("train_set", {}).get("value", []))
train_set = [tuple(t) for t in train_set]

all_tuples = list(combinations(range( d ), k))
test_set = [t for t in all_tuples if t not in train_set]

test_set = random.sample(test_set, min(500,len(test_set))  )
batch_size = 500
# Initialize a dictionary to store the aggregated results
all_results = defaultdict(lambda: [0] * (k + 1))
key_counts = defaultdict(int)

base_path = "/scratch/bbjr/abedsol1/parity_best_models/"
# addressess_cl = [(10,base_path+"model_gpt2_context_length10_nlayers3_nheads1_ntrain_tasks81train_d15k3_epoch_160.pt"),
#                  (20,base_path+"model_gpt2_context_length20_nlayers3_nheads1_ntrain_tasks81train_d15k3_epoch_100.pt"),
#                  (30,base_path+"model_gpt2_context_length30_nlayers3_nheads1_ntrain_tasks81train_d15k3_epoch_80.pt"),
#                  (40,base_path+"model_gpt2_context_length40_nlayers3_nheads1_ntrain_tasks81train_d15k3_epoch_80.pt")]

if k==4:
 cls = [20,30,40]
else:
    cls = [10,20,30,40]

for cl in [args.cl]:

    target_pattern = re.compile(
        rf"model_gpt2_context_length{cl}_nlayers3_nheads1_ntrain_tasks81train_d15k{k}_epoch_\d+\.pt"
    )

    # Search for matching files in the folder
    for file_name in os.listdir(base_path):
        print(file_name)
        if target_pattern.match(file_name):
            model_path = base_path + file_name
            break

    checkpoint_path = model_path
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint, strict=True)

    x = []
    y = []

    for i in tqdm(range(0, len(test_set), batch_size), desc="Processing Batches"):


        test_set_batch = test_set[i:i+batch_size]
        # print(test_set_batch)

        # ipdb.set_trace()

        # Create datasets
        val_out_diff_seq_dataset = create_dataset(
            d=d, train_task_keys=test_set_batch, N=cl, use_cot=True, selected_samples=remaining_sequences
        )

        dataloader_val = DataLoader(val_out_diff_seq_dataset, batch_size=256, shuffle=True)

        # Evaluate the model
        model.to(device)
        result = evaluate_model(model, dataloader_val, device, k=k)

        # Log the final averaged bar plots
        # for key, coord_accuracies in result['accuracy_per_key_per_coord'].items():
        #     wandb.log({
        #         f"final_accuracy_per_coor_{key}": wandb.plot.bar(
        #             wandb.Table(data=[[j, coord_accuracies[j]] for j in range(k + 1)],
        #                         columns=["Coordinate", "Accuracy"]),
        #             "Coordinate", "Accuracy",
        #             title=f"Final Accuracy for Key {key}"
        #         )
        #     })

        # ipdb.set_trace()

        data = result['accuracy_per_key']

        # Prepare data for plotting
        x = x + [str(key) for key in data.keys()]   # Convert tuples to strings for x-axis
        y = y + list(data.values())  # Accuracy values for y-axis

        del data

    # Calculate average
    # After collecting x and y
    # Flatten y to a single list

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import numpy as np


    # Calculate the average of y
    y_avg = np.mean(y)

    # Dynamic figure size
    num_x = len(x)


    # Create the figure and axes
    fig = Figure(figsize=(20, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    # Create the bar plot
    bars = ax.bar(range(num_x), y, color='skyblue', edgecolor='black')  # Use integer indices for bars

    # Add a horizontal line for the average
    ax.axhline(y=y_avg, color='red', linestyle='--', linewidth=1.5, label=f'Average: {y_avg:.2f}')

    # Set x-ticks and x-tick labels
    x_labels = [str(item) for item in x]  # Convert tuples to strings
    ax.set_xticks(range(num_x))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)

    # Add labels, title, and legend
    ax.set_xlabel('Categories', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title('Bar Plot with Average Line', fontsize=14)
    ax.legend()

    # Save the figure to a file
    canvas.print_figure(f"bar_plot_d{d}_k{k}_cl{cl}.png", dpi=300)

    # Create the bar chart
    # fig = go.Figure()

    # # Add bars
    # fig.add_trace(go.Bar(x=x, y=y, text=y, textposition='auto', name="Accuracy"))
    #
    # # Add average line
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=[avg_y] * len(x),  # Repeat average value for all x values
    #     mode='lines',
    #     name=f"Average = {avg_y:.2f}",
    #     line=dict(color='red', dash='dash')
    # ))
    #
    # # Update layout
    # fig.update_layout(
    #     title=f"Accuracy Per Tuple CL={cl}(with Average)",
    #     xaxis_title="Tuples",
    #     yaxis_title="Accuracy",
    #     xaxis_tickangle=-45
    # )
    #
    # # Log the figure to WandB
    # wandb.log({f"Accuracy Per Tuple CL={cl}": fig})
    #
    #
    # # Close the plot to avoid overlap in subsequent plots
    # plt.close()






    # ipdb.set_trace()

# Finish the wandb run
run.finish()
