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
from mesa.time import SimultaneousActivation
from MASAgentCNN import ProcessorAgent
from trainMASCNN import get_model
import torch
import random
import time
import numpy as np

from partition_utils import make_dirichlet_cifar10_indices

class ParallelizationModel(Model):
    def __init__(self, Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args):
        super().__init__()
        self.device = device
        self.args = args
        self.num_processors = args.processors
        self.schedule = SimultaneousActivation(self)
        self.total_comm_cost = 0
        # Compute-capacity upper bound (min fixed at 1.0)
        self.capacity_max = getattr(self.args, "capacity_max", 2.0)

    ###
        self.total_processing_time = 0.0
        self.total_e2e_time = 0.0  # (optional but recommended) track end-to-end time cleanly

        # IMPORTANT: Use args.num_classes (UA/EV-DETRAC = 3). Keep a safe fallback for older runs.
        num_classes = getattr(self.args, "num_classes", 100 if args.ds == "CIFAR100" else 10)

        # Initialize model architecture (UA/EV-DETRAC uses args.num_classes)
        pretrained = getattr(self.args, "imagenet_pretrained", False)
        self.model = get_model(self.args.arch, num_classes=num_classes, pretrained=pretrained).to(device)



    ###
        self._split_data_and_create_agents(Training_ds, Training_lbls, Testing_ds, Testing_lbls)

    ### replaced
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

        # --- UNIFORM compute capacities in [1.0, capacity_max] ---
        low, high = 1.0, float(self.capacity_max)
        capacities = [random.uniform(low, high) for _ in range(self.num_processors)]
        self.capacities = capacities

        if dataset_mode:
            # Shard indices (not the dataset itself)
            train_indices = list(range(total_train))
            test_indices = list(range(total_test))

            train_per_agent = total_train // self.num_processors
            test_per_agent = total_test // self.num_processors

            for i in range(self.num_processors):
                train_start = i * train_per_agent
                train_end = (i + 1) * train_per_agent if i < self.num_processors - 1 else total_train

                test_start = i * test_per_agent
                test_end = (i + 1) * test_per_agent if i < self.num_processors - 1 else total_test

                agent = ProcessorAgent(
                    unique_id=i,
                    model=self,
                    Training_ds=Training_ds,                 # dataset object
                    Training_lbls=train_indices[train_start:train_end],  # indices shard
                    Testing_ds=Testing_ds,                   # dataset object
                    Testing_lbls=test_indices[test_start:test_end],      # indices shard
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



    def synchronize_weights(self):
        if len(self.schedule.agents) == 1:
            print("Only one processor: skipping weight synchronization and communication cost.")
            return 0

        num_agents = len(self.schedule.agents)
        ref_agent = self.schedule.agents[0]
        model_state_dict = {k: v.cpu() for k, v in ref_agent.neural_net_model.state_dict().items()}

        model_size = sum(v.element_size() * v.nelement() for v in model_state_dict.values())
        epoch_comm_cost = sum(getattr(a, "last_cc", 0) for a in self.schedule.agents)

        self.total_comm_cost += epoch_comm_cost
        print(f"Communication cost this epoch: {epoch_comm_cost} bytes; "
              f"Cumulative cost: {self.total_comm_cost} bytes")

        global_weights = {k: torch.zeros_like(v) for k, v in model_state_dict.items()}
        for agent in self.schedule.agents:
            if agent.neural_net_model is None:
                continue
            state_dict = {k: v.cpu() for k, v in agent.neural_net_model.state_dict().items()}
            for k, v in state_dict.items():
                global_weights[k] += v

        for k in global_weights:
            global_weights[k] = global_weights[k].float() / num_agents

        m, n = self.args.latency
        if m * n != 0:
            latency = random.uniform(m / 1000, n / 1000)
            print(f"Simulating network latency of {latency:.4f} seconds during weight synchronization...")
            time.sleep(latency)

        for agent in self.schedule.agents:
            if agent.neural_net_model is None:
                continue
            agent.neural_net_model.load_state_dict(global_weights)
        # Compute comm time in seconds from this epoch's cost
        bw_Bps = self.args.net_bw_mbps * 125000.0  # Mbps → bytes/sec
        comm_time_s = epoch_comm_cost / bw_Bps

        # Store the comm time in each agent so run_model() can read it
        for agent in self.schedule.agents:
            agent.last_comm_time_s = comm_time_s
        return epoch_comm_cost

    def step(self):
        self.schedule.step()

        comm_cost = self.synchronize_weights()

        total_correct = 0
        total_test_examples = 0
        slowest_processing_time = 0

        for agent in self.schedule.agents:
            total_correct += agent.correct_classifications
            total_test_examples += agent.test_set_size
            slowest_processing_time = max(slowest_processing_time, agent.processing_time)

        total_testing_accuracy = (total_correct / total_test_examples) * 100 if total_test_examples > 0 else 0

        print(f"Total Testing Accuracy this epoch: {total_correct}/{total_test_examples} = {total_testing_accuracy:.2f}%")
        print(f"Processing Time (compute only) this epoch: {slowest_processing_time:.4f} seconds")

        # NEW: comm time from agents (per-epoch, per-node → take the slowest)
        slowest_comm_time = max(getattr(a, "last_comm_time_s", 0.0) for a in self.schedule.agents)
        print(f"Communication Time (assumed {self.args.net_bw_mbps} Mbps): {slowest_comm_time:.4f} seconds")

        end_to_end_time = slowest_processing_time + slowest_comm_time
        print(f"Processing Time (end-to-end) this epoch: {end_to_end_time:.4f} seconds")

        self.total_processing_time += slowest_processing_time  # keep compute-only total if you want
        # (Optional) also keep a total E2E time:
        self.total_e2e_time = getattr(self, "total_e2e_time", 0.0) + end_to_end_time

        print(f"Cumulative Communication Cost until now: {self.total_comm_cost} bytes")

    def run_model(self, num_steps):
        """
        Run SYNC distributed training for num_steps GLOBAL epochs.

        IMPORTANT:
        - Do NOT call agent.step() directly here.
        - self.step() is the authoritative per-epoch method because it:
            (1) steps the Mesa schedule (agents do 1 local epoch)
            (2) calls synchronize_weights() (global averaging + comm accounting)
            (3) prints epoch metrics and updates totals
        """

        # GRAND totals (test)
        total_correct_items = 0
        total_test_items = 0

        # GRAND totals (training)
        total_train_correct = 0
        total_train_items = 0

        for epoch in range(1, num_steps + 1):
            print(f"\n** Epoch {epoch} of {num_steps} starts **")

            # Run one GLOBAL epoch (local training + sync + logging)
            self.step()

            # ---- Aggregate accuracy snapshots from agents (after this epoch) ----
            correct_this_epoch = sum(a.correct_classifications for a in self.schedule.agents)
            test_this_epoch = sum(a.test_set_size for a in self.schedule.agents)

            train_correct_this_epoch = sum(getattr(a, "train_correct", 0) for a in self.schedule.agents)
            train_total_this_epoch = sum(getattr(a, "train_total", 0) for a in self.schedule.agents)

            total_correct_items += correct_this_epoch
            total_test_items += test_this_epoch
            total_train_correct += train_correct_this_epoch
            total_train_items += train_total_this_epoch

            train_acc = (100.0 * train_correct_this_epoch / train_total_this_epoch) if train_total_this_epoch else 0.0
            print(f"TOTAL Training Accuracy after epoch: {train_correct_this_epoch}/{train_total_this_epoch} = {train_acc:.2f}%")

        grand_test_acc = (100.0 * total_correct_items / total_test_items) if total_test_items else 0.0
        print(f"\nGRAND TOTAL Final Accuracy: {total_correct_items}/{total_test_items} = {grand_test_acc:.2f}%")

        print(f"GRAND TOTAL Communication Cost over all epochs: {getattr(self, 'total_comm_cost', 0)} bytes")
        print(f"GRAND TOTAL Processing Time (compute only): {getattr(self, 'total_processing_time', 0.0):.4f} seconds")
        print(f"GRAND TOTAL Processing end-to-end time: {getattr(self, 'total_e2e_time', 0.0):.4f} seconds")

        grand_train_acc = (100.0 * total_train_correct / total_train_items) if total_train_items else 0.0
        print(f"GRAND TOTAL Training Accuracy: {total_train_correct}/{total_train_items} = {grand_train_acc:.2f}%")

