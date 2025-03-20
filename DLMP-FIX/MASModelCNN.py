# MASModelCNN.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from mesa import Model
from mesa.time import SimultaneousActivation
from MASAgentCNN import ProcessorAgent  # Ensure this module is up-to-date

class ParallelizationModel(Model):
    def __init__(self, X_train, y_train, X_test, y_test, device, args):
        super().__init__()
        self.num_processors = args.processors
        self.Training_ds = X_train
        self.Training_lbls = y_train
        self.Testing_ds = X_test
        self.Testing_lbls = y_test
        self.device = device
        self.args = args
        self.schedule = SimultaneousActivation(self)  # All agents act simultaneously

        # Split the training and testing data among processors (nodes)
        self._split_data_and_create_agents()

    def _split_data_and_create_agents(self):
        split_Training_ds = np.array_split(self.Training_ds, self.num_processors)
        split_Training_lbls = np.array_split(self.Training_lbls, self.num_processors)
        split_Testing_ds = np.array_split(self.Testing_ds, self.num_processors)
        split_Testing_lbls = np.array_split(self.Testing_lbls, self.num_processors)

        for i in range(self.num_processors):
            # Pass parameters positionally to match ProcessorAgent's __init__ signature.
            agent = ProcessorAgent(
                i,
                self,
                split_Training_ds[i],
                split_Training_lbls[i],
                split_Testing_ds[i],
                split_Testing_lbls[i],
                self.device,
                self.args
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
        latency = np.random.uniform(m / 1000, n / 1000)  # Latency in seconds
        print(f"Simulating network latency of {latency:.4f} seconds during weight synchronization...")
        time.sleep(latency)  # Delay to simulate communication delay

        # Average the weights
        for k in global_weights:
            global_weights[k] /= num_agents

        # Update each agent's model with the averaged weights
        for agent in self.schedule.agents:
            if agent.neural_net_model is not None:
                agent.neural_net_model.load_state_dict(global_weights)

    def step(self):
        """Advance the model by one step and evaluate overall performance."""
        self.schedule.step()

        total_correct = 0
        total_test_examples = 0
        slowest_processing_time = 0

        for agent in self.schedule.agents:
            total_correct += agent.correct_classifications
            total_test_examples += agent.test_set_size
            slowest_processing_time = max(slowest_processing_time, agent.processing_time)

        total_testing_accuracy = (total_correct / total_test_examples) * 100
        print(f"Total Testing Accuracy: {total_correct}/{total_test_examples} = {total_testing_accuracy:.2f}%")
        print(f"Slowest Processing Time Across Nodes: {slowest_processing_time:.4f} seconds")

