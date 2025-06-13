from mesa import Model
from mesa.time import RandomActivation
from MASAAgentCNN import ProcessorAgent
from trainMASACNN import get_model
import torch
import random
import time

class ParallelizationModel(Model):
    """
    Simulates a parallel processing environment where multiple processor agents
    train on different data partitions and exchange their neural network weights
    asynchronously using peer-to-peer communication.
    """
    def __init__(self, Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args):
        super().__init__()
        self.device = device
        self.args = args
        self.num_processors = args.processors
        self.schedule = RandomActivation(self)
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

        # Assign peers
        for agent in self.schedule.agents:
            agent.set_peers([a for a in self.schedule.agents if a != agent])

    def step(self):
        """
        Executes one simulation step.
        Aggregates performance metrics from all agents after each step.
        """
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

    def run_model(self, epochs):
        """
        Runs the simulation for the specified number of epochs.
        """
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            self.step()
        print(f"\nGRAND TOTAL Communication Cost over all epochs: {self.total_comm_cost} bytes")
