from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW,SGD
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from transformers import set_seed
import os
from data.OOD import create_dataset
import ipdb
from tqdm import tqdm
import torch
import torch.nn.functional as F
import random
import wandb
import numpy as np
from itertools import combinations

from datetime import datetime
import os
import argparse
import itertools
import random


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

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Description of your script.")

# Add arguments
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--n_heads', type=int, default=3)
parser.add_argument('--context_length', type=int, default=1)
parser.add_argument('--data_method', type=str, default="n_train_task")
parser.add_argument('--model_type', type=str, default="gpt")
parser.add_argument('--pre_trained', type=int, default=0)
parser.add_argument('--dim_ambient', type=int, default=10)
parser.add_argument('--dim_k', type=int, default=3)
parser.add_argument('--split_type', type=int, default=1)
parser.add_argument('--gpt2_width', type=int, default=192)
parser.add_argument('--max_lr', type=float, default=6e-5)
parser.add_argument('--train_task_perc', type=float, default=0.8)

parser.add_argument('--n_train_tasks', type=int, default=1)
parser.add_argument('--n_train_seqs_exponent', type=int, default=9)
parser.add_argument('--n_test_seqs_exponent', type=int, default=9)
parser.add_argument('--positional_encoding', type=str, default="default")
parser.add_argument('--reg_position_embeddings', type=int, default=0)
parser.add_argument('--n_train_per_task', type=int, default=1000)

parser.add_argument('--n_train_total', type=int, default=1000)


parser.add_argument('--cot', type=int, default=0)








# Parse arguments
args = parser.parse_args()


wandb_init = {}
wandb_init["project_name"] = "parity_fixed_d_k"
wandb_init["mode"] = 'online'
wandb_init["key"] = "1840d187a4e8cfe03c2d82a3ff679f732d94b304"
wandb_init["org"] = "belkinlab"

os.environ["WANDB_API_KEY"] = wandb_init['key']
os.environ["WANDB_MODE"] = wandb_init['mode']  # online or offline
run = wandb.init(project=wandb_init['project_name'], entity=wandb_init['org'])

def set_all_seeds(seed):
    # Python random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hugging Face Transformers
    set_seed(seed)
    # Environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate_model(model, val_loader,  device, k=3, cot = 0):
    """
    Evaluate the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        val_loader: DataLoader for the validation set.
        criterion: Loss function used during training.
        device: The device ('cpu' or 'cuda') to perform evaluation on.

    Returns:
        result: Dictionary containing average validation loss and accuracies:
                {
                    "avg_loss": Average validation loss,
                    "accuracy_all": Accuracy when all last k+1 coordinates match,
                    "accuracy_0": Accuracy for the last coordinate,
                    "accuracy_1": Accuracy for the second-to-last coordinate,
                    ...
                    "accuracy_k": Accuracy for the k-th coordinate.
                }
    """
    model.eval()  # Set the model to evaluation mode
    correct_all = 0
    correct_coords = [0] * (k + 1)
    total = 0

    with torch.no_grad():  # Disable gradient computation

        for batch in tqdm(val_loader, desc="Validation Progress", unit="batch"):

            inputs, targets = batch['input_ids'], batch['labels']
            inputs, targets = inputs.to(device), targets.to(device)

            shifted_targets = targets[:, 1:]  # Ignore the first token
            inputs = inputs[:, :-1]  # Remove the last token from inputs

            # Forward pass
            outputs = model(input_ids=inputs)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)


            logits = torch.argmax(logits, dim=-1)  # Predicted tokens


            # filtered_logits = logits[:, -k - 1:]
            # filtered_targets = shifted_targets[:, -k - 1:]



            # Compare each coordinate and the full match
            if cot:
                filtered_logits = logits[:, - (k+1):]
                filtered_targets = shifted_targets[:, - (k+1):]
                matches_all = torch.all(filtered_logits == filtered_targets, dim=1)  # All coordinates match
                correct_all += matches_all.sum().item()

                for i in range(k+1 ): #origianlly it was k+1
                    matches_coord = (filtered_logits[:, i] == filtered_targets[:, i])
                    correct_coords[i] += matches_coord.sum().item()

            else:
                filtered_logits = logits[:, - 1:]
                filtered_targets = shifted_targets[:, - 1:]
                matches_all = torch.all(filtered_logits == filtered_targets, dim=1)  # All coordinates match
                correct_all += matches_all.sum().item()

            total += inputs.size(0)

        # Calculate metrics
    # avg_loss = val_loss / total  # Average loss
    accuracy_all = correct_all / total  # Accuracy for all coordinates

    if cot:
        accuracy_coords = [correct / total for correct in correct_coords]

        # Prepare results
        result = { "accuracy_all": accuracy_all}
        for i in range(k + 1):
            result[f"accuracy_{i}"] = accuracy_coords[i]
    else:
        result = {"accuracy_all": accuracy_all}

    return result



# Example usage
SEED = 42
set_all_seeds(SEED)
d = args.dim_ambient
k = args.dim_k
#ICL data non overlaping
N = args.context_length  # Number of sequences concatenated per example
# use_cot = True
# n_train_each = args.n_train_per_task  # Fixed value for each tuple

vocab_size = 2
tuples = np.array(list(combinations(range(d), k)))



np.random.seed(42)
random.seed(42)
all_tuples = list(combinations(range( d ), k))



if args.data_method == "dlogd":

    import math


    train_set = sorted(list(set(random.sample( all_tuples, int( 2*d*math.log(d) ) ) ))) #+ [tuple(range(i, i + k)) for i in range( d - k + 1)]
    test_set = [t for t in all_tuples if t not in train_set]
    test_set = random.sample(test_set, min(len(test_set), 200))

else:

    test_set = random.sample(all_tuples,min(200,len(all_tuples) -args.n_train_tasks ) )
    train_set = [t for t in all_tuples if t not in test_set]
    train_set = random.sample(train_set, int(args.n_train_tasks) )

np.random.seed(42)
random.seed(42)
# Generate all possible sequences of size d
# all_sequences = list(itertools.product([0, 1], repeat=d))
# # Randomly select m sequences
# # selected_sequences = random.sample(all_sequences, 2**(d-1) )
# selected_sequences = random.sample(all_sequences, 2**args.n_train_seqs_exponent )
# # Get the remaining sequences
# remaining_sequences = [seq for seq in all_sequences if seq not in selected_sequences]
# remaining_sequences = random.sample(remaining_sequences, 2**args.n_test_seqs_exponent )

selected_sequences = None
remaining_sequences = None

# ipdb.set_trace()

if args.cot:

    n_train_per_task = args.n_train_total//len(train_set)

    val_out_diff_seq_dataset = create_dataset(
        d=d, train_task_keys=test_set, N=N, use_cot=True, selected_samples=remaining_sequences, num_samples= n_train_per_task
    )


    train_dataset = create_dataset(
        d=d, train_task_keys=train_set, N=N, use_cot=True, selected_samples=selected_sequences, num_samples= n_train_per_task
    )
    # val_in_dataset = create_dataset(
    #     d=d, train_task_keys=train_set, N=N, use_cot=True, selected_samples=remaining_sequences, num_samples= args.n_train_per_task
    # )
    #
    # val_out_same_seq_dataset = create_dataset(
    #     d=d, train_task_keys=test_set, N=N, use_cot=True, selected_samples=selected_sequences, num_samples= args.n_train_per_task
    # )



else:

    train_dataset = create_dataset(
        d=d, train_task_keys=train_set, N=N, use_cot=False, selected_samples = selected_sequences
    )
    val_in_dataset = create_dataset(
        d=d, train_task_keys=train_set, N=N, use_cot=False, selected_samples = remaining_sequences
    )

    val_out_same_seq_dataset = create_dataset(
        d=d, train_task_keys=test_set, N=N, use_cot=False, selected_samples = selected_sequences
    )

    val_out_diff_seq_dataset = create_dataset(
        d=d, train_task_keys=test_set, N=N, use_cot=False, selected_samples = remaining_sequences
    )






train_size = len(train_set)

run.config.update({
    "test_set": test_set,
    "context_length": N,
    "train_set": train_set,
    "n_train_tasks": train_size,
    # "train_task_perc":args.train_task_perc,
    # "n_train_seqs_exponent ":args.n_train_seqs_exponent,
    "train_seqs":selected_sequences,
    # "n_test_seqs_exponent":args.n_test_seqs_exponent,
    "test_seqs":remaining_sequences,
    # "n_train_per_task":args.n_train_per_task,
    "n_train_total":args.n_train_total,
})
name_data = f"d:{d}--k:{k} --n_train_tasks:{train_size} -- context_length = {N}"




dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) #default 64
# dataloader_val_in = DataLoader(val_in_dataset, batch_size=64, shuffle=True)
# dataloader_val_out_same_seq_dataset = DataLoader(val_out_same_seq_dataset, batch_size=64, shuffle=True)
dataloader_val_out_diff_seq_dataset = DataLoader(val_out_diff_seq_dataset, batch_size=64, shuffle=True)
print("data is ready")


# Define GPT-2 configuration with small vocabulary and your desired sequence length





config = GPT2Config(
    vocab_size=2,  # Binary tokens: 0 and 1
    n_embd=args.gpt2_width,    # Embedding dimension (default GPT-2)
    n_layer=args.n_layers,    # Number of transformer layers
    n_head=args.n_heads,     # Number of attention heads
    n_positions=2048 # Maximum sequence length
)
# Initialize GPT-2 model
model = GPT2LMHeadModel(config)
optimizer = AdamW(model.parameters(), lr=args.max_lr)  # default=6e-5# GPT-2 optimizer




name_model = f"model:{args.model_type}_nlayers:{args.n_layers}_nheads:{args.n_heads}"
run.name = name_model + name_data

run.config.update({
    "n_layers":args.n_layers,
    "n_heads":args.n_heads,
    "dim_ambient":args.dim_ambient,
    "dim_k":args.dim_k,
    "gpt2_width":args.gpt2_width,
    "positional encoding":args.positional_encoding,
    "cot":args.cot,
    "data_method":args.data_method,
})




criterion = nn.CrossEntropyLoss(ignore_index = -100)  # For next-token prediction

# optimizer = SGD(model.parameters(), lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
from torch.optim.lr_scheduler import CosineAnnealingLR
epochs = 8000 #5000 # Number of training epochs
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) #defualt: eta_min=1e-6
step = 0
accuracy_test_best = 0


for epoch in range(epochs):

    # print(f"epoch{epoch}")
    model.train()
    total_loss = 0
    correct = 0
    n_data = 0

    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
        for batch in progress_bar:

            # ipdb.set_trace()
            inputs, targets = batch['input_ids'], batch['labels']
            inputs, targets = inputs.to(device), targets.to(device)

            # ipdb.set_trace()

            # Shift targets for next-token prediction
            shifted_targets = targets[:, 1:]  # Ignore the first token
            shifted_inputs = inputs[:, :-1]  # Remove the last token from inputs

            # Forward pass
            # ipdb.set_trace()
            outputs = model(input_ids=shifted_inputs)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)



            # Compute loss using reshape instead of view
            loss = criterion(logits.reshape(-1, vocab_size), shifted_targets.reshape(-1))
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.shape[0]

            # Create a mask to identify valid (non-dummy) positions
            # valid_mask = shifted_targets != -100  # True for valid positions, False for dummies
            logits = torch.argmax(logits, dim=-1)

            # ipdb.set_trace()

            # Extract only the last k+1 coordinates
            if args.cot:
                filtered_logits = logits[:, - (k+1):]
                filtered_targets = shifted_targets[:, - (k+1):]
                matches = torch.all(filtered_logits == filtered_targets, dim=1)
            else:
                filtered_logits =  logits[:, - 1:]
                filtered_targets =shifted_targets[:, - 1:]
                matches = torch.all(filtered_logits == filtered_targets, dim=1)

            # ipdb.set_trace()

            # Update counters
            n_data += shifted_targets.shape[0]
            correct_count = matches.sum().item()
            correct += correct_count

            # Increment the step counter
            step += 1

    # Step the scheduler after each epoch
    scheduler.step()

    # ipdb.set_trace()
    avg_loss = total_loss / n_data
    accuracy_train = correct / n_data
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, accuracy:{accuracy_train}")
    # avg_loss_test, accuracy_test = evaluate_model(model, dataloader_val, criterion, device, k=k)

    if (epoch + 1) % 3 == 0:
        # result_in = evaluate_model(model, dataloader_val_in, device, k=k, cot = args.cot)
        # result_out_same_seq_dataset = evaluate_model(model, dataloader_val_out_same_seq_dataset, device, k=k, cot = args.cot)
        result_out_diff_seq_dataset = evaluate_model(model, dataloader_val_out_diff_seq_dataset, device, k=k, cot = args.cot)

        accuracy_test = result_out_diff_seq_dataset["accuracy_all"]
        print(f"validation Epoch {epoch + 1}/{epochs},  accuracy:{accuracy_test:.4f}")

        current_lr = optimizer.param_groups[0]['lr']

        run.log({ "epoch":epoch,
                 "learning rate": current_lr,
                 "train accuracy": accuracy_train,
                 "train loss":avg_loss,
                 "model":args.model_type,
                 # ** {f"val_in {key}": value for key, value in result_in.items()},
                 # **{f"val_same_seq_out {key}": value for key, value in result_out_same_seq_dataset.items()},
                  **{f"val_diff_seq_out {key}": value for key, value in result_out_diff_seq_dataset.items()}


                 })


        if accuracy_test_best< accuracy_test:

            accuracy_test_best = accuracy_test
            # Get current day and month
            current_time = datetime.now()
            day = current_time.day
            month = current_time.month
            hour = current_time.hour
            # Create folder path using day and month
            folder_path = f"/scratch/bbjr/abedsol1/ICL_parity_saved_models/" \
                          f"{args.data_method}_d{d}_k{k}/cot_{args.cot}"


            # Create the folder if it does not exist
            os.makedirs(folder_path, exist_ok=True)

            # Construct the save path within the folder
            save_path = os.path.join(
                folder_path,
                f"d{d}_k{k}_model_{args.model_type}_context_length{N}_nlayers{args.n_layers}"
                f"_nheads{args.n_heads}_ntrain_tasks{train_size}_ntrain_task{args.n_train_tasks}"
                f"n_train_total{args.n_train_total}_epoch_{epoch + 1}_width{args.gpt2_width}.pt"
            )
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")


