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
        print(f"--- Total Accuracy: {total_correct}/{total_processed} = {final_accuracy:.2f}% ---")

        # Calculate and print Testing Accuracy
        self.report_testing_accuracy()

    def synchronize_weights(self):
        """
        Synchronize weights among all agents by averaging them.
        This method ensures that each node has the same set of weights.
        """
        global_weights = None
        num_agents = len(self.schedule.agents)

        # Collect and average weights
        for agent in self.schedule.agents:
            # Access the neural network model stored in each agent
            model_weights = agent.neural_net_model.state_dict()
            if global_weights is None:
                global_weights = {k: torch.zeros_like(v) for k, v in model_weights.items()}

            for k, v in model_weights.items():
                global_weights[k] += v

        # Average weights
        for k in global_weights:
            global_weights[k] /= num_agents

        # Update each agent's model with the averaged weights
        for agent in self.schedule.agents:
            agent.neural_net_model.load_state_dict(global_weights)

    def report_testing_accuracy(self):
        """
        Evaluate the synchronized model on the testing dataset and report accuracy.
        """
        # Assuming that all agents now have the same synchronized model
        test_model = self.schedule.agents[0].neural_net_model

        # Convert test data to torch tensors and move to device
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.int64).to(self.device)

        # Run inference on the test data
        with torch.no_grad():
            test_outputs = test_model(X_test_tensor)
            correct_classifications = (test_outputs.argmax(1) == y_test_tensor).sum().item()
            total_examples = len(y_test_tensor)

            testing_accuracy = (correct_classifications / total_examples) * 100

        # Print Testing Accuracy in the specified format
        print(f"--- Testing Accuracy: {correct_classifications}/{total_examples} = {testing_accuracy:.2f}% ---")
