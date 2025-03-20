# MASModelCNN.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from mesa import Model
from mesa.time import SimultaneousActivation
from MASAgentCNN import ProcessorAgent
from trainMASCNN import train_simulated

class ParallelizationModel(Model):
    def __init__(self, Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args):
        super().__init__()
        self.num_processors = args.processors
        self.Training_ds = Training_ds
        self.Training_lbls = Training_lbls
        self.Testing_ds = Testing_ds
        self.Testing_lbls = Testing_lbls
        self.device = device
        self.args = args
        self.schedule = SimultaneousActivation(self)
        self._split_data_and_create_agents()

    def _split_data_and_create_agents(self):
        split_Training_ds = np.array_split(self.Training_ds, self.num_processors)
        split_Training_lbls = np.array_split(self.Training_lbls, self.num_processors)
        split_Testing_ds = np.array_split(self.Testing_ds, self.num_processors)
        split_Testing_lbls = np.array_split(self.Testing_lbls, self.num_processors)

        for i in range(self.num_processors):
            agent = ProcessorAgent(
                unique_id=i,
                model=self,
                Training_ds=split_Training_ds[i],
                Training_lbls=split_Training_lbls[i],
                Testing_ds=split_Testing_ds[i],
                Testing_lbls=split_Testing_lbls[i],
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

        # Collect and sum weights from all agents
        for agent in self.schedule.agents:
            if agent.neural_net_model is not None:
                for k, v in agent.neural_net_model.state_dict().items():
                    global_weights[k] += v

        # Simulate realistic network latency based on the --latency argument (m, n)
        m, n = self.args.latency
        if n - m != 0:
            latency = np.random.uniform(m / 1000, n / 1000)  # Latency in seconds
            print(f"Simulating network latency of {latency:.4f} seconds during weight synchronization...")
            time.sleep(latency)

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
            total_correct += agent.correct_classifications
            total_test_examples += agent.test_set_size
            slowest_processing_time = max(slowest_processing_time, agent.processing_time)

        total_testing_accuracy = (total_correct / total_test_examples) * 100

        print(f"Total Testing Accuracy: {total_correct}/{total_test_examples} = {total_testing_accuracy:.2f}%")
        print(f"Slowest Processing Time Across Nodes: {slowest_processing_time:.4f} seconds")

    def _advance_time(self):
        # This method is required by Mesa's scheduler. Here we simply pass.
        pass

