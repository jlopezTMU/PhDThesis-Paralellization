"""
DLMP: Deep Learning Multi-Processing Simulator

Author: Jorge A. Lopez
Affiliation: Toronto Metropolitan University

Description:
Agent-based simulation framework for studying coordination strategies
in distributed deep learning systems.

Supports non-IDD distribution

Repository:
https://github.com/DLMPsim/DLMP

License:
MIT License
"""

from mesa import Model
from mesa.time import RandomActivation
from MASAAgentCNN import ProcessorAgent
from trainMASACNN import get_model
import torch
import random
import time
import numpy as np

from partition_utils import make_dirichlet_cifar10_indices, make_dirichlet_uadetrac_indices

class ParallelizationModel(Model):
    """
    Simulates a parallel processing environment where multiple processor agents
    train on different data partitions and exchange their neural network weights
    asynchronously using a ring topology.
    """
    

    def __init__(self, Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args):
        super().__init__()
        self.device = device
        self.args = args
        self.num_processors = args.processors
        self.schedule = RandomActivation(self)
        self.total_comm_cost = 0
        # Compute-capacity upper bound (min is fixed at 1.0)
        self.capacity_max = getattr(self.args, "capacity_max", 2.0)

        # Initialize model architecture (UA-DETRAC uses args.num_classes)
        num_classes = getattr(self.args, "num_classes", 100 if self.args.ds == "CIFAR100" else 10)
        pretrained = getattr(self.args, "imagenet_pretrained", False)
        self.model = get_model(self.args.arch, num_classes=num_classes, pretrained=pretrained).to(device)


        # Create agents and split data
        self._split_data_and_create_agents(Training_ds, Training_lbls, Testing_ds, Testing_lbls)
        # Assign ring peers (one neighbor each)
            
        agents = self.schedule.agents
        n = len(agents)
        for i, agent in enumerate(agents):
            if n > 1:
                neighbor = agents[(i + 1) % n]
                agent.set_peers([neighbor])
            else:
                agent.set_peers([])  # no peer in single-node runs → zero CC/time        
      
    def _split_data_and_create_agents(self, Training_ds, Training_lbls, Testing_ds, Testing_lbls):
        """
        Supports two modes:
        1) Array-mode (MNIST/CIFAR): Training_ds is numpy/array-like and is sliced as before.
        2) Dataset-mode (UA_DETRAC): Training_ds is a torch Dataset; we shard by indices and stream via DataLoader in train_simulated().
        """
        import random
        from torch.utils.data import Dataset as TorchDataset

        dataset_mode = isinstance(Training_ds, TorchDataset) and isinstance(Testing_ds, TorchDataset)

        total_train = len(Training_ds)
        total_test = len(Testing_ds)

        # Assign compute capacities uniformly in [1.0, capacity_max].
        low, high = 1.0, float(self.capacity_max)
        capacities = [random.uniform(low, high) for _ in range(self.num_processors)]
        self.capacities = capacities

        if dataset_mode:
            # Shard indices (not the dataset itself)
            test_indices = list(range(total_test))

            partition_mode = getattr(self.args, "partition", "iid")

            if partition_mode == "nonIID_uadetrac":
                if self.args.ds != "UA_DETRACnonIID":
                    raise ValueError("nonIID_uadetrac is only supported for UA_DETRACnonIID.")

                train_labels = np.asarray(Training_ds.labels)

                train_shards = make_dirichlet_uadetrac_indices(
                    train_labels,
                    self.num_processors,
                    alpha=getattr(self.args, "dirichlet_alpha", 0.5),
                    seed=getattr(self.args, "partition_seed", 42)
                )

            else:
                train_indices = list(range(total_train))
                train_per_agent = total_train // self.num_processors
                train_shards = []

                for i in range(self.num_processors):
                    train_start = i * train_per_agent
                    train_end = (i + 1) * train_per_agent if i < self.num_processors - 1 else total_train
                    train_shards.append(train_indices[train_start:train_end])

            test_per_agent = total_test // self.num_processors

            for i in range(self.num_processors):
                test_start = i * test_per_agent
                test_end = (i + 1) * test_per_agent if i < self.num_processors - 1 else total_test

                agent = ProcessorAgent(
                    unique_id=i,
                    model=self,
                    Training_ds=Training_ds,                 # dataset object
                    Training_lbls=train_shards[i],           # index shard
                    Testing_ds=Testing_ds,                   # dataset object
                    Testing_lbls=test_indices[test_start:test_end],      # test index shard
                    device=self.device,
                    args=self.args
                )
                self.schedule.add(agent)

                agent.compute_capacity = capacities[i]
                print(f"Node {i + 1} compute capacity (uniform {low}..{high}): {agent.compute_capacity:.3f}")

        else:
            ###

            partition_mode = getattr(self.args, "partition", "iid")

            if partition_mode == "nonIID_cifar10":
                if self.args.ds != "CIFAR10":
                    raise ValueError("nonIID_cifar10 is only supported for CIFAR10.")
                
                train_shards = make_dirichlet_cifar10_indices(
                    Training_lbls,
                    self.num_processors,
                    alpha=getattr(self.args, "dirichlet_alpha", 0.5),
                    seed=getattr(self.args, "partition_seed", 42)
)

            else:
                train_per_agent = total_train // self.num_processors
                train_shards = []

                for i in range(self.num_processors):
                    train_start = i * train_per_agent

                    if i < self.num_processors - 1:
                        train_end = (i + 1) * train_per_agent
                    else:
                        train_end = total_train

                    train_shards.append(
                        np.arange(train_start, train_end)
                    )

            test_per_agent = total_test // self.num_processors

            for i in range(self.num_processors):
                train_idx = train_shards[i]

                test_start = i * test_per_agent

                if i < self.num_processors - 1:
                    test_end = (i + 1) * test_per_agent
                else:
                    test_end = total_test

                agent = ProcessorAgent(
                    unique_id=i,
                    model=self,
                    Training_ds=Training_ds[train_idx],
                    Training_lbls=np.asarray(Training_lbls)[train_idx],
                    Testing_ds=Testing_ds[test_start:test_end],
                    Testing_lbls=Testing_lbls[test_start:test_end],
                    device=self.device,
                    args=self.args
                )

                self.schedule.add(agent)

                agent.compute_capacity = capacities[i]

                print(
                    f"Node {i + 1} compute capacity "
                    f"(uniform {low}..{high}): "
                    f"{agent.compute_capacity:.3f}"
                )
            ###
            

    def step(self):
        self.schedule.step()
        total_correct = 0
        total_test_examples = 0
        slowest_processing_time = 0
        epoch_cc = 0
        for agent in self.schedule.agents:
            total_correct += agent.correct_classifications
            total_test_examples += agent.test_set_size
            slowest_processing_time = max(slowest_processing_time, agent.processing_time)
            epoch_cc += agent.last_cc
        self.total_comm_cost += epoch_cc
        total_testing_accuracy = (total_correct / total_test_examples) * 100 if total_test_examples > 0 else 0
        print(f"Total Testing Accuracy this epoch: {total_correct}/{total_test_examples} = {total_testing_accuracy:.2f}%")
        self.last_epoch_total_acc = total_testing_accuracy
        # Compute-only time (slowest node)
        print(f"Processing Time (compute only) this epoch: {slowest_processing_time:.4f} seconds")
        self.total_compute_time = getattr(self, "total_compute_time", 0.0) + slowest_processing_time

        # Communication time for the epoch (slowest node).
        slowest_comm_time = max(getattr(a, 'last_comm_time_s', 0.0) for a in self.schedule.agents)
        print(f"Communication Time (assumed {self.args.net_bw_mbps} Mbps): {slowest_comm_time:.4f} seconds")
        self.total_comm_time = getattr(self, "total_comm_time", 0.0) + slowest_comm_time
        # end-to-end = compute + comm
        end_to_end_time = slowest_processing_time + slowest_comm_time
        print(f"Processing Time (end-to-end) this epoch: {end_to_end_time:.4f} seconds")
        print(f"Cumulative Communication Cost until now: {self.total_comm_cost} bytes")
        self.total_e2e_time = getattr(self, "total_e2e_time", 0.0) + end_to_end_time

    def run_model(self, epochs):
    
        total_train_correct = 0
        total_train_items   = 0

        total_comm_cost = 0
        total_compute_time = 0.0
        total_comm_time = 0.0
        total_e2e_time = 0.0
        total_correct_items = 0
        total_test_items = 0

        for epoch_idx in range(1, epochs + 1):
            print(f"\n** Epoch {epoch_idx} of {epochs} starts **")

            # Reset per-epoch totals
            slowest_processing_time = 0.0
            slowest_comm_time = 0.0
            total_correct = 0
            total_test_examples = 0

            if self.num_processors == 1:
                # No sync or comm time in single-node runs
                for agent in self.schedule.agents:
                    agent.last_cc = 0
                    agent.last_comm_time_s = 0.0
                self.step()
            else:
                self.step()

            # After agents step, collect metrics
            for agent in self.schedule.agents:
                total_correct += agent.correct_classifications
                total_test_examples += agent.test_set_size
                slowest_processing_time = max(slowest_processing_time, agent.processing_time)
                slowest_comm_time = max(slowest_comm_time, getattr(agent, "last_comm_time_s", 0.0))

            # Communication cost (skip if single node)
            if self.num_processors > 1:
                comm_cost = sum(getattr(agent, "last_cc", 0) for agent in self.schedule.agents)
                total_comm_cost += comm_cost
                print(f"Communication cost this epoch: {comm_cost} bytes; "
                      f"Cumulative cost: {total_comm_cost} bytes")
            else:
                print("Only one processor: skipping weight synchronization and communication cost.")

            # Accuracy and times for this epoch
            accuracy = 100.0 * total_correct / total_test_examples
            total_correct_items += total_correct
            total_test_items += total_test_examples
            total_compute_time += slowest_processing_time
                       
            print(f"TOTAL Final Accuracy after epoch: {accuracy:.2f}%")
            print(f"Processing Time (compute only): {slowest_processing_time:.4f} seconds")

            epoch_comm_time_sum = (0.0 if self.num_processors == 1 else
                sum(getattr(agent, "last_comm_time_s", 0.0) for agent in self.schedule.agents))
            total_comm_time += epoch_comm_time_sum
            print(f"Communication Time (sum over nodes): {epoch_comm_time_sum:.4f} seconds")
            total_e2e_time += slowest_processing_time + slowest_comm_time
            print(f"Processing end-to-end time: {slowest_processing_time + slowest_comm_time:.4f} seconds")
            # --- TOTAL Training Accuracy after epoch (all nodes, all datasets) ---
            train_correct_this_epoch = sum(a.train_correct for a in self.schedule.agents)
            train_total_this_epoch   = sum(a.train_total   for a in self.schedule.agents)
            train_accuracy = 100.0 * train_correct_this_epoch / train_total_this_epoch if train_total_this_epoch else 0.0
            print(f"TOTAL Training Accuracy after epoch: {train_correct_this_epoch}/{train_total_this_epoch} = {train_accuracy:.2f}%")

            # accumulate GRAND totals
            total_train_correct += train_correct_this_epoch
            total_train_items   += train_total_this_epoch

        # --- Final GRAND TOTALS ---
        grand_test_acc = 100.0 * total_correct / total_test_examples if total_test_examples else 0.0
        print(f"FINAL Test Accuracy: {total_correct}/{total_test_examples} = {grand_test_acc:.2f}%")
        print(f"GRAND TOTAL Communication Cost over all epochs: {total_comm_cost} bytes")
        print(f"GRAND TOTAL Communication Time over all epochs: {total_comm_time:.4f} seconds")
        print(f"GRAND TOTAL Processing Time (compute only): {total_compute_time:.4f} seconds")
        print(f"GRAND TOTAL Processing end-to-end time: {total_e2e_time:.4f} seconds")

        grand_train_acc = 100.0 * train_correct_this_epoch / train_total_this_epoch if train_total_this_epoch else 0.0
        print(f"FINAL Training Accuracy: {train_correct_this_epoch}/{train_total_this_epoch} = {grand_train_acc:.2f}%")
             
