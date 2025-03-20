# MASAgentCNN.py
from mesa import Agent
import time
from trainMASCNN import train_simulated
import torch
import numpy as np

class ProcessorAgent(Agent):
    def __init__(self, unique_id, model,
                 Training_ds, Training_lbls,
                 Testing_ds, Testing_lbls,
                 device, args):
        # Call the parent class initializer
        Agent.__init__(self, unique_id, model)

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

    def step(self):
        start_time = time.time()

        def sync_callback(model_state_dict):
            self.model.synchronize_weights(model_state_dict)

        # Train using this agent's dataset subset.
        # Note: We now pass len(self.Training_ds) as the original_training_size argument.
        result = train_simulated(
            self.unique_id,
            list(range(len(self.Training_ds))),   # Train indices
            list(range(len(self.Testing_ds))),    # Validation indices
            self.Training_ds,
            self.Training_lbls,
            self.device,
            self.args,
            len(self.Training_ds),                # original_training_size
            sync_callback
        )

        (self.fold_idx_loss,
         self.fold_idx_accuracy,
         self.neural_net_model,
         correct_classifications_train,
         correct_test,
         test_size,
         processing_time) = result

        end_time = time.time()
        self.processing_time = end_time - start_time

        self.correct_classifications = correct_test
        self.test_set_size = test_size

        # Accuracy = correct classifications / test set size
        accuracy = (self.correct_classifications / self.test_set_size) * 100
        num_processors = self.model.num_processors

        print(f"--- Node {self.unique_id + 1}/{num_processors} completed in {self.processing_time:.4f} seconds ---")
        print(f"--- Node {self.unique_id + 1} Accuracy: {self.correct_classifications}/{self.test_set_size} = {accuracy:.2f}% ---")

