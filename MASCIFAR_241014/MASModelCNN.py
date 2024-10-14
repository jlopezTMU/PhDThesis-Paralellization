from mesa import Model
from mesa.time import SimultaneousActivation  # Import SimultaneousActivation scheduler
from MASAgentCNN import ProcessorAgent  # Import ProcessorAgent from the MASAgentCNN module
import numpy as np
import torch
import time
import random

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

        # Split the training data among processors (nodes)
        self._split_data_and_create_agents()

    def _split_data_and_create_agents(self):
        split_X_train = np.array_split(self.X_train, self.num_processors)
        split_y_train = np.array_split(self.y_train, self.num_processors)
        split_X_test = np.array_split(self.X_test, self.num_processors)
        split_y_test = np.array_split(self.y_test, self.num_processors)

        for i in range(self.num_processors):
            agent = ProcessorAgent(
                unique_id=i,
                model=self,
                X_train=split_X_train[i],
                y_train=split_y_train[i],
                X_test=split_X_test[i],
                y_test=split_y_test[i],
                device=self.device,
                args=self.args
            )
            self.schedule.add(agent)

    def synchronize_weights(self, model_state_dict=None):
        """Synchronize weights among all agents by averaging them."""
        if model_state_dict is None:
            return  # Nothing to synchronize if no state dict is provided

        global_weights = {k: torch.zeros_like(v) for k, v in model_state_dict.items()}
        num_agents = len(self.schedule.agents)

        # Collect and average weights from all agents
        for agent in self.schedule.agents:
            if agent.neural_net_model is not None:
                for k, v in agent.neural_net_model.state_dict().items():
                    global_weights[k] += v

        # Simulate realistic network latency based on the --latency argument (m, n)
        m, n = self.args.latency
        latency = random.uniform(m / 1000, n / 1000)  # Latency in seconds
        print(f"Simulating network latency of {latency:.4f} seconds during weight synchronization...")
        time.sleep(latency)  # Add a delay to simulate network communication delay

        # Average the weights
        for k in global_weights:
            global_weights[k] /= num_agents

        # Update each agent's model with the averaged weights
        for agent in self.schedule.agents:
            if agent.neural_net_model is not None:
                agent.neural_net_model.load_state_dict(global_weights)

    def step(self):
        self.schedule.step()

        total_correct = 0
        total_test_examples = 0
        slowest_processing_time = 0

        for agent in self.schedule.agents:
            # Sum the correct classifications and test set sizes across all nodes
            total_correct += agent.correct_classifications
            total_test_examples += agent.test_set_size
            slowest_processing_time = max(slowest_processing_time, agent.processing_time)

        # Calculate total testing accuracy: total correct classifications / total test examples across all nodes
        total_testing_accuracy = (total_correct / total_test_examples) * 100

        # Print the results
        print(f"Total Testing Accuracy: {total_correct}/{total_test_examples} = {total_testing_accuracy:.2f}%")
        print(f"Slowest Processing Time Across Nodes: {slowest_processing_time:.4f} seconds")

