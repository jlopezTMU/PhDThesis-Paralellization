import numpy as np
import torch
import time
import random
from mesa import Model
from mesa.time import SimultaneousActivation
from MASAgentCNN import ProcessorAgent

class ParallelizationModel(Model):
    def __init__(self, Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args):
        super(ParallelizationModel, self).__init__()
        self.num_processors = args.processors
        self.Training_ds = Training_ds
        self.Training_lbls = Training_lbls
        self.Testing_ds = Testing_ds
        self.Testing_lbls = Testing_lbls
        self.device = device
        self.args = args
        # Initialize cumulative communication cost (in bytes) as grand total over all epochs
        self.total_comm_cost = 0
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
        if model_state_dict is None:
            return  # Nothing to synchronize if no state dict is provided

        # Ensure the provided state dict is on CPU.
        model_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}

        # *** Calculate communication cost ***
        # Compute the cost (in bytes) of transmitting one complete model.
        single_comm_cost = sum(v.element_size() * v.nelement() for v in model_state_dict.values())
        # Assuming each agent transmits the same amount, multiply by the number of agents.
        num_agents = len(self.schedule.agents)
        epoch_comm_cost = num_agents * single_comm_cost
        # Accumulate epoch cost into the grand total.
        self.total_comm_cost += epoch_comm_cost
        print(f"Communication cost this epoch: {epoch_comm_cost} bytes; Cumulative cost: {self.total_comm_cost} bytes")
        # ************************************

        # Proceed with weight synchronization: create global_weights by summing each agent's weights.
        global_weights = {k: torch.zeros_like(v) for k, v in model_state_dict.items()}
        for agent in self.schedule.agents:
            if agent.neural_net_model is None:
                continue  # Skip agents without a model.
            if isinstance(agent.neural_net_model, dict):
                state_dict = {k: v.cpu() for k, v in agent.neural_net_model.items()}
            else:
                state_dict = {k: v.cpu() for k, v in agent.neural_net_model.state_dict().items()}
            for k, v in state_dict.items():
                global_weights[k] += v

        for k in global_weights:
            global_weights[k] /= num_agents

        # Simulate network latency if specified.
        m, n = self.args.latency
        if m * n != 0:
            latency = random.uniform(m / 1000, n / 1000)
            print(f"Simulating network latency of {latency:.4f} seconds during weight synchronization...")
            time.sleep(latency)

        # Update each agent's model (or state dict) with the averaged weights.
        for agent in self.schedule.agents:
            if agent.neural_net_model is None:
                continue
            if isinstance(agent.neural_net_model, dict):
                agent.neural_net_model = global_weights
            else:
                agent.neural_net_model.load_state_dict(global_weights)

    def step(self):
        # Execute one simulation step (representing one epoch).
        self.schedule.step()

        total_correct = 0
        total_test_examples = 0
        slowest_processing_time = 0

        for agent in self.schedule.agents:
            total_correct += agent.correct_classifications
            total_test_examples += agent.test_set_size
            slowest_processing_time = max(slowest_processing_time, agent.processing_time)

        total_testing_accuracy = (total_correct / total_test_examples) * 100 if total_test_examples > 0 else 0
        print(f"Total Testing Accuracy this epoch: {total_correct}/{total_test_examples} = {total_testing_accuracy:.2f}%")
        print(f"Slowest Processing Time this epoch: {slowest_processing_time:.4f} seconds")
        print(f"Cumulative Communication Cost until now: {self.total_comm_cost} bytes")

    def run_simulation(self, epochs):
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            self.step()
        print(f"\nGRAND TOTAL Communication Cost over all epochs: {self.total_comm_cost} bytes")
