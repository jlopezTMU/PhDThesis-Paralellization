import torch
import argparse
from MASModelCNN import ParallelizationModel
from sklearn.model_selection import train_test_split  # Import train_test_split for dataset splitting

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Simulated parallel processing on MNIST using MAS")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--processors', type=int, default=4, help='Number of simulated processors (nodes) to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')

    args = parser.parse_args()

    # Check if GPU is available and requested
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("***Using device: GPU***")
    else:
        device = torch.device("cpu")
        print("***Using device: CPU***")

    # Load MNIST dataset
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Convert the dataset to numpy arrays, keeping the 2D shape for convolution operations
    X = mnist_dataset.data.numpy().reshape(-1, 1, 28, 28) / 255.0
    y = mnist_dataset.targets.numpy()

    # Split the dataset into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Adjust behavior based on the number of processors
    if args.processors == 1:
        # Use the entire dataset for training (48000 examples) when only one processor is specified
        X_train = X_train
        y_train = y_train

    # Initialize the Parallelization Model
    model = ParallelizationModel(X_train, y_train, X_test, y_test, device, args)

    # Run the simulation for a single step (can be extended to multiple steps)
    model.step()

if __name__ == '__main__':
    main()
(myenv) PS C:\Users\georg\OneDrive\phd\Thesis\MASDL\MASOK>
(myenv) PS C:\Users\georg\OneDrive\phd\Thesis\MASDL\MASOK>
(myenv) PS C:\Users\georg\OneDrive\phd\Thesis\MASDL\MASOK>
(myenv) PS C:\Users\georg\OneDrive\phd\Thesis\MASDL\MASOK> ls


    Directory: C:\Users\georg\OneDrive\phd\Thesis\MASDL\MASOK


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a---l        2024-08-28   6:26 PM           4651 GuiMASCNN.py
-a---l        2024-08-28   3:08 PM           2268 mainMASCNN.py
-a---l        2024-08-28   5:17 PM           2222 MASAgentCNN.py
-a---l        2024-08-28   5:25 PM           6369 MASModelCNN.py
-a---l        2024-08-28   5:20 PM           3246 trainMASCNN.py


(myenv) PS C:\Users\georg\OneDrive\phd\Thesis\MASDL\MASOK> cat MASAgentCNN.py
from mesa import Agent
import time
from trainMASCNN import train_simulated

class ProcessorAgent(Agent):
    def __init__(self, unique_id, model, X_train, y_train, X_test, y_test, device, args):
        super().__init__(unique_id, model)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
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

        # Calculate accuracy as correctly classified examples / total examples processed
        accuracy = self.correct_classifications / self.total_examples_processed * 100

        # Access num_processors from the ParallelizationModel instance
        num_processors = self.model.num_processors

        # Print in the specified format: X/Y=Z%
        print(f"--- Node {self.unique_id + 1}/{num_processors} completed in {self.processing_time:.4f} seconds ---")
        print(f"--- Node {self.unique_id + 1} Accuracy: {self.correct_classifications}/{self.total_examples_processed} = {accuracy:.2f}% ---")
(myenv) PS C:\Users\georg\OneDrive\phd\Thesis\MASDL\MASOK> ls


    Directory: C:\Users\georg\OneDrive\phd\Thesis\MASDL\MASOK


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a---l        2024-08-28   6:26 PM           4651 GuiMASCNN.py
-a---l        2024-08-28   3:08 PM           2268 mainMASCNN.py
-a---l        2024-08-28   5:17 PM           2222 MASAgentCNN.py
-a---l        2024-08-28   5:25 PM           6369 MASModelCNN.py
-a---l        2024-08-28   5:20 PM           3246 trainMASCNN.py


(myenv) PS C:\Users\georg\OneDrive\phd\Thesis\MASDL\MASOK> cat MASModelCNN.py
from mesa import Model
from mesa.time import SimultaneousActivation
from MASAgentCNN import ProcessorAgent
import numpy as np
import torch

class ParallelizationModel(Model):
    def __init__(self, X_train, y_train, X_test, y_test, device, args):
        super().__init__()
        self.num_processors = args.processors
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = device
        self.args = args
        self.schedule = SimultaneousActivation(self)

        # Print to confirm dataset sizes
        print(f"Initializing with {self.X_train.shape[0]} training examples")
        print(f"Initializing with {self.X_test.shape[0]} testing examples")

        # Split training data into `num_processors` parts
        self._split_data_and_create_agents()

    def _split_data_and_create_agents(self):
        """
        Split the dataset into n parts according to the number of processors.
        If there is only one processor, don't split the data.
        """
        if self.num_processors == 1:
            # Use the entire dataset without splitting
            print(f"Using the entire dataset for a single processor: {self.X_train.shape[0]} training examples")
            agent = ProcessorAgent(0, self, self.X_train, self.y_train, self.X_test, self.y_test, self.device, self.args)
            self.schedule.add(agent)
        else:
            # Split training data into `num_processors` parts
            split_X_train = np.array_split(self.X_train, self.num_processors)
            split_y_train = np.array_split(self.y_train, self.num_processors)

            for i in range(self.num_processors):
                print(f"Node {i+1} using {split_X_train[i].shape[0]} training examples")
                agent = ProcessorAgent(
                    unique_id=i,
                    model=self,
                    X_train=split_X_train[i],
                    y_train=split_y_train[i],
                    X_test=self.X_test,
                    y_test=self.y_test,
                    device=self.device,
                    args=self.args
                )
                self.schedule.add(agent)

    def step(self):
        self.schedule.step()  # Each agent runs its training step

        # Synchronize weights across all processors after each step
        if self.num_processors > 1:
            self.synchronize_weights()

        # Aggregate results
        losses = [agent.fold_loss for agent in self.schedule.agents]
        processing_times = [agent.processing_time for agent in self.schedule.agents]
        correct_classifications = [agent.correct_classifications for agent in self.schedule.agents]
        total_examples_processed = [agent.total_examples_processed for agent in self.schedule.agents]

        average_loss = sum(losses) / len(losses)
        cumulative_time = sum(processing_times)
        average_time_per_node = cumulative_time / len(processing_times)
        total_correct = sum(correct_classifications)
        total_processed = sum(total_examples_processed)

        final_accuracy = (total_correct / total_processed) * 100

        # Print final cumulative statistics in the specified format
        print(f"--- Cumulative processing time for all nodes: {cumulative_time:.4f} seconds ---")
        print(f"--- Average processing time per node: {average_time_per_node:.4f} seconds ---")
        print(f"--- Total Accuracy (All nodes): {total_correct}/{total_processed} = {final_accuracy:.2f}% ---")

        # Calculate and print Testing Accuracy
        self.report_testing_accuracy()

    def synchronize_weights(self, model_state_dict=None):
        """
        Synchronize weights among all agents by averaging them.
        This method ensures that each node has the same set of weights.
        If model_state_dict is provided, it is used for synchronization.
        """
        if model_state_dict is None:
            return  # Nothing to synchronize if no state dict is provided

        global_weights = {k: torch.zeros_like(v) for k, v in model_state_dict.items()}
        num_agents = len(self.schedule.agents)

        # Collect and average weights from non-None agents
        for agent in self.schedule.agents:
            if agent.neural_net_model is not None:
                for k, v in agent.neural_net_model.state_dict().items():
                    global_weights[k] += v

        # Average weights
        for k in global_weights:
            global_weights[k] /= num_agents

        # Update each agent's model with the averaged weights
        for agent in self.schedule.agents:
            if agent.neural_net_model is not None:
                agent.neural_net_model.load_state_dict(global_weights)

    def report_testing_accuracy(self):
        """
        Evaluate the synchronized model on the testing dataset and report accuracy.
        """
        # Perform a final synchronization to ensure all models are the same
        self.synchronize_weights(self.schedule.agents[0].neural_net_model.state_dict())

        # Use the first agent's model as the representative synchronized model
        test_model = self.schedule.agents[0].neural_net_model

        # Set the model to evaluation mode to disable dropout, etc.
        test_model.eval()

        # Convert test data to torch tensors and move to device
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.int64).to(self.device)

        # Reshape the input tensor to [batch_size, 1, 28, 28]
        X_test_tensor = X_test_tensor.view(-1, 1, 28, 28)

        # Run inference on the test data
        with torch.no_grad():
            test_outputs = test_model(X_test_tensor)
            correct_classifications = (test_outputs.argmax(1) == y_test_tensor).sum().item()
            total_examples = len(y_test_tensor)

            testing_accuracy = (correct_classifications / total_examples) * 100

        # Print Testing Accuracy in the specified format
        ## This is the one that is not useful print(f"--- Testing Accuracy: {correct_classifications}/{total_examples} = {testing_accuracy:.2f}% ---")
