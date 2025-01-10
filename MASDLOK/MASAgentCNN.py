from mesa import Agent
import time
from trainMASCNN import train_simulated
import torch

class ProcessorAgent(Agent):
    def __init__(self, unique_id, model, Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args):
        super().__init__(unique_id, model)
        self.Training_ds = Training_ds
        self.Training_lbls = Training_lbls
        self.Testing_ds = Testing_ds
        self.Testing_lbls = Testing_lbls
        self.device = device
        self.args = args
        self.fold_idx_loss = None
        self.fold_idx_accuracy = None
        self.neural_net_model = None  # Reference to the neural network model
        self.processing_time = None
        self.correct_classifications = 0  # Correct classifications for the node's test dataset portion
        self.test_set_size = len(self.Testing_ds)  # Size of the node's test dataset portion
    # Represents the work performed by an individual agent (node) in one step of the simulation.
    def step(self):
        start_time = time.time() #records the time to measure how long the step takes

        def sync_callback(model_state_dict):
            self.model.synchronize_weights(model_state_dict)

        # Properly unpack the return values from `train_simulated` (trains the nn using the agent's portion of the dataset)
        result = train_simulated(
            self.unique_id,
            list(range(len(self.Training_ds))),  # Train indices
            list(range(len(self.Testing_ds))),   # Validation indices (if used)
            self.Training_ds,
            self.Training_lbls,
            self.device,
            self.args,
            len(self.Training_ds),  # Pass the original training size
            sync_callback  # Pass the sync callback to handle weight synchronization
        )

        # Unpack values based on the actual return structure
        (self.fold_idx_loss, self.fold_idx_accuracy, self.neural_net_model,
         correct_classifications_train, correct_val, val_set_size,
         self.processing_time) = result

        end_time = time.time()
        self.processing_time = end_time - start_time

        # Store the correct classifications for this node's test portion
        self.correct_classifications = correct_val
        self.test_set_size = val_set_size  # Should match len(self.Testing_ds)

        # Calculate node accuracy: correct classifications / total examples in the node's test set
        accuracy = self.correct_classifications / self.test_set_size * 100

        # Access num_processors from the ParallelizationModel instance
        num_processors = self.model.num_processors

        # Print in the specified format: X/Y=Z%
        print(f"--- Node {self.unique_id + 1}/{num_processors} completed in {self.processing_time:.4f} seconds ---")
        print(f"--- Node {self.unique_id + 1} Accuracy: {self.correct_classifications}/{self.test_set_size} = {accuracy:.2f}% ---")



