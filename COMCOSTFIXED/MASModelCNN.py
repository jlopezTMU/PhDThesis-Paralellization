from mesa import Model
from mesa.time import SimultaneousActivation
from MASAgentCNN import ProcessorAgent
from trainMASCNN import get_model
import torch
import random
import time

class ParallelizationModel(Model):
    """
    Simulates a parallel processing environment where multiple processor agents
    train on different data partitions and synchronize their neural network weights.
    """
    def __init__(self, Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args):
        super().__init__()
        self.device = device
        self.args = args
        self.num_processors = args.processors
        self.schedule = SimultaneousActivation(self)
        self.total_comm_cost = 0  # Initialize cumulative communication cost

        # Create the shared (global) neural network model for copying
        self.model = get_model(self.args.arch, num_classes=10).to(device)

        self._split_data_and_create_agents(Training_ds, Training_lbls, Testing_ds, Testing_lbls)

    def _split_data_and_create_agents(self, Training_ds, Training_lbls, Testing_ds, Testing_lbls):
        """
        Splits the training and testing datasets among agents and creates each ProcessorAgent.
        """
        total_train = len(Training_ds)
        total_test = len(Testing_ds)

        # Calculate the number of examples per agent
        train_per_agent = total_train // self.num_processors
        test_per_agent = total_test // self.num_processors

        for i in range(self.num_processors):
            train_start = i * train_per_agent
            train_end = (i + 1) * train_per_agent if i < self.num_processors - 1 else total_train

            test_start = i * test_per_agent
            test_end = (i + 1) * test_per_agent if i < self.num_processors - 1 else total_test

            agent = ProcessorAgent(
                unique_id=i,
                model=self,  # Pass the simulation model (parent) to the agent
                Training_ds=Training_ds[train_start:train_end],
                Training_lbls=Training_lbls[train_start:train_end],
                Testing_ds=Testing_ds[test_start:test_end],
                Testing_lbls=Testing_lbls[test_start:test_end],
                device=self.device,
                args=self.args
            )
            self.schedule.add(agent)

    def synchronize_weights(self, model_state_dict=None):
        """
        Averages model weights from all agents and simulates the communication cost
        as well as potential network latency during weight synchronization.
        """
        if model_state_dict is None:
            return  # Nothing to synchronize

        # Ensure all tensors are on CPU for consistent processing
        model_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}
        num_agents = len(self.schedule.agents)

        # Calculate communication cost (bytes) of transmitting one complete model
        single_comm_cost = sum(v.element_size() * v.nelement() for v in model_state_dict.values())
        epoch_comm_cost = num_agents * single_comm_cost
        self.total_comm_cost += epoch_comm_cost

        print(f"Communication cost this epoch: {epoch_comm_cost} bytes; "
              f"Cumulative cost: {self.total_comm_cost} bytes")

        # Initialize a dictionary to hold the sum of weights from each agent
        global_weights = {k: torch.zeros_like(v) for k, v in model_state_dict.items()}
        for agent in self.schedule.agents:
            if agent.neural_net_model is None:
                continue  # Skip agents without a model yet
            state_dict = {k: v.cpu() for k, v in agent.neural_net_model.state_dict().items()}
            for k, v in state_dict.items():
                global_weights[k] += v

        # Average the weights across all agents
        for k in global_weights:
            global_weights[k] /= num_agents

        # Simulate network latency if configured (args.latency should be a tuple: (m, n))
        m, n = self.args.latency
        if m * n != 0:
            latency = random.uniform(m / 1000, n / 1000)
            print(f"Simulating network latency of {latency:.4f} seconds during weight synchronization...")
            time.sleep(latency)

        # Update each agent's model with the averaged weights
        for agent in self.schedule.agents:
            if agent.neural_net_model is None:
                continue
            agent.neural_net_model.load_state_dict(global_weights)

    def step(self):
        """
        Executes one simulation step (representing one epoch).
        Aggregates performance metrics from all agents after each step.
        """
        self.schedule.step()

        total_correct = 0
        total_test_examples = 0
        slowest_processing_time = 0

        # Combine metrics from all agents
        for agent in self.schedule.agents:
            total_correct += agent.correct_classifications
            total_test_examples += agent.test_set_size
            slowest_processing_time = max(slowest_processing_time, agent.processing_time)

        total_testing_accuracy = (total_correct / total_test_examples) * 100 if total_test_examples > 0 else 0

        print(f"Total Testing Accuracy this epoch: {total_correct}/{total_test_examples} = {total_testing_accuracy:.2f}%")
        print(f"Slowest Processing Time this epoch: {slowest_processing_time:.4f} seconds")
        print(f"Cumulative Communication Cost until now: {self.total_comm_cost} bytes")

    def run_model(self, epochs):
        """
        Runs the simulation for the specified number of epochs.
        """
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            self.step()
        print(f"\nGRAND TOTAL Communication Cost over all epochs: {self.total_comm_cost} bytes")
