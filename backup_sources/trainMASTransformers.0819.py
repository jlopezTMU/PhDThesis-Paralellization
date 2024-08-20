import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
import os
import sys
import logging

# Suppress all warnings by redirecting output to null
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Re-enable output after your training code
def restore_output():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# Set logging level to suppress warnings
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('pytorch').setLevel(logging.ERROR)

def train_simulated(fold, train_idx, val_idx, X, y, device, args):
    # Load the pre-trained BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1).to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Prepare the training data subset using list comprehensions
    train_texts = [X[i] for i in train_idx]
    val_texts = [X[i] for i in val_idx]
    y_train_fold = torch.tensor([y[i] for i in train_idx], dtype=torch.float32).to(device)
    y_val_fold = torch.tensor([y[i] for i in val_idx], dtype=torch.float32).to(device)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

    # Move encodings to device
    train_encodings = {key: val.to(device) for key, val in train_encodings.items()}
    val_encodings = {key: val.to(device) for key, val in val_encodings.items()}

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Initialize scaler for mixed precision training
    scaler = GradScaler()

    # Define the number of accumulation steps
    accumulation_steps = 4  # Adjust this number based on your GPU memory capacity

    # Training loop with gradient accumulation
    model.train()
    total_correct = 0  # Track total correct predictions
    total_examples = 0  # Track total number of examples

    all_predicted_labels = []
    all_batch_labels = []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()  # Reset gradients at the start of each epoch

        # Accumulate gradients over several batches
        for i, _ in enumerate(range(0, len(train_encodings['input_ids']), args.batch_size)):
            # Slice the batch data
            batch_start = i * args.batch_size
            batch_end = batch_start + args.batch_size
            batch_input_ids = train_encodings['input_ids'][batch_start:batch_end]
            batch_attention_mask = train_encodings['attention_mask'][batch_start:batch_end]
            batch_labels = y_train_fold[batch_start:batch_end]

            # Forward pass with mixed precision
            with autocast():
                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
                loss = outputs.loss / accumulation_steps  # Divide loss by accumulation steps

            # Backward pass
            scaler.scale(loss).backward()

            # Perform optimizer step and reset gradients only after accumulating the specified number of steps
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Debug: print shapes and check for mismatched shapes
            predicted_labels = (outputs.logits > 0.5).float().squeeze()
            ## no need print(f"    Debug: Predicted Labels Shape: {predicted_labels.shape}, Batch Labels Shape: {batch_labels.shape}")

            # Append predictions and labels for later aggregation
            all_predicted_labels.append(predicted_labels)
            all_batch_labels.append(batch_labels)

            # Print batch progress
            if (i + 1) % 25 == 0:  # Print every 25 batches
                print(f"    Batch {i+1}: Loss = {loss.item():.4f}")

            ## Debug: print shapes and check for mismatched shapes
            #predicted_labels = (outputs.logits > 0.5).float().squeeze()
            #print(f"    Debug: Predicted Labels Shape: {predicted_labels.shape}, Batch Labels Shape: {batch_labels.shape}")

            # Calculate number of correct predictions
            correct = (predicted_labels == batch_labels).sum().item()
            total_correct += correct
            total_examples += batch_labels.size(0)

            # Debug: print the correct predictions for this batch
            # no need print(f"    Debug: Batch Correct = {correct}, Total Correct = {total_correct}, Batch Size = {batch_labels.size(0)}")

        # If the number of batches isn't a multiple of accumulation_steps, perform one last step
        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Aggregate predictions and labels across all processors
        aggregated_predicted_labels = torch.cat(all_predicted_labels)
        aggregated_batch_labels = torch.cat(all_batch_labels)

        # Calculate and print training accuracy for this epoch
        correct = (aggregated_predicted_labels == aggregated_batch_labels).sum().item()
        total_correct += correct
        total_examples += aggregated_batch_labels.size(0)
        training_accuracy = (total_correct / total_examples) * 100
        print(f"Finished Epoch {epoch+1}/{args.epochs} - Training Accuracy: {training_accuracy:.2f}%")

        # Reset the lists for the next epoch
        all_predicted_labels = []
        all_batch_labels = []

    # Validation loop
    model.eval()
    with torch.no_grad():
        with autocast():
            val_outputs = model(**val_encodings, labels=y_val_fold)
            val_loss = val_outputs.loss.item()
            val_accuracy = ((val_outputs.logits > 0.5).float().squeeze() == y_val_fold).float().mean().item()

    return val_loss, val_accuracy, model

# Restore output after your code execution
restore_output()

root@d948903bbf3b:~/sandbox/MASDL#
