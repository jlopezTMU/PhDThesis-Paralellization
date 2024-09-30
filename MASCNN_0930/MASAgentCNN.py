from mesa import Agent
import time
from trainMASCNN import train_simulated
import torch

class ProcessorAgent(Agent):
    def __init__(self, unique_id, model, X_train, y_train, X_test, y_test, device, args):
        super().__init__(unique_id, model)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test  # Assign test data
        self.y_test = y_test  # Assign test labels
        self.device = device
        self.args = args
        self.fold_loss = None
        self.fold_accuracy = None
        self.neural_net_model = None  # Reference to the LeNet model
        self.processing_time = None
        self.correct_classifications = 0
        self.total_examples_processed = len(X_train)  # Track total number of examples processed

    def step(self):
        start_time = time.time()

        def sync_callback(model_state_dict):
            self.model.synchronize_weights(model_state_dict)

        # Run training and store the trained model
        self.fold_loss, self.fold_accuracy, self.neural_net_model, self.correct_classifications = train_simulated(
            self.unique_id, 
            list(range(len(self.X_train))),  # Train indices
            list(range(len(self.X_test))),   # Validation indices (if used)
            self.X_train, 
            self.y_train, 
            self.device, 
            self.args, 
            len(self.X_train),  # Pass the original training size
            sync_callback  # Pass the sync callback to handle weight synchronization
        )

        end_time = time.time()
        self.processing_time = end_time - start_time

        # Calculate training accuracy
        accuracy = self.correct_classifications / self.total_examples_processed * 100

        # Access num_processors from the ParallelizationModel instance
        num_processors = self.model.num_processors

        # Print training processing time and accuracy per node
        print(f"--- Node {self.unique_id + 1}/{num_processors} completed in {self.processing_time:.4f} seconds ---")
        print(f"--- Node {self.unique_id + 1} Training Accuracy: {self.correct_classifications}/{self.total_examples_processed} = {accuracy:.2f}% ---")

        # After training and synchronization, compute testing accuracy per node
        # Prepare test data
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.int64).to(self.device)
        # Reshape input tensor to [batch_size, 1, 28, 28]
        X_test_tensor = X_test_tensor.view(-1, 1, 28, 28)

        self.neural_net_model.eval()
        with torch.no_grad():
            test_outputs = self.neural_net_model(X_test_tensor)
            correct_test = (test_outputs.argmax(1) == y_test_tensor).sum().item()
            test_accuracy = (correct_test / len(y_test_tensor)) * 100
            print(f"--- Node {self.unique_id + 1} Testing Accuracy: {correct_test}/{len(y_test_tensor)} = {test_accuracy:.2f}% ---")
