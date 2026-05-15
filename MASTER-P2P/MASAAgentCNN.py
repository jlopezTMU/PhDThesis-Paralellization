"""
DLMP: Deep Learning Multi-Processing Simulator

Author: Jorge A. Lopez
Affiliation: Toronto Metropolitan University

Description:
Agent-based simulation framework for studying coordination strategies
in distributed deep learning systems.

This version includes the averaging of weights for the descentralized P2P-ring algorithm

Repository:
https://github.com/DLMPsim/DLMP

License:
MIT License
"""
import copy
from mesa import Agent
from trainMASACNN import train_simulated, build_loaders

import torch


class ProcessorAgent(Agent):
    """
    Represents a simulated processor (node) that trains on a subset of the data.
    Each agent maintains a local copy of the neural network and participates
    in peer-to-peer weight exchange.
    """
    def __init__(self, unique_id, model, Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args):
        super().__init__(unique_id, model)
        self.Training_ds = Training_ds
        self.Training_lbls = Training_lbls
        self.Testing_ds = Testing_ds
        self.Testing_lbls = Testing_lbls
        self.device = device
        self.args = args
        # Build DataLoaders once per agent to avoid recreating them each epoch.
        self.train_loader, self.test_loader, self.dataset_mode, self.array_eval = build_loaders(
            self.Training_ds, self.Training_lbls,
            self.Testing_ds, self.Testing_lbls,
            self.device, self.args
        )

        self.compute_capacity = 1.0  # Baseline compute capacity; agents may simulate slower nodes but not faster ones.


        self.fold_idx_loss = None
        self.fold_idx_accuracy = None
        self.processing_time = 0.0
 
        self.correct_classifications = 0
        # In dataset mode, Testing_lbls is a list of indices for the agent shard.
        self.test_set_size = len(self.Testing_lbls) if self.Testing_lbls is not None else len(self.Testing_ds)


        self.neural_net_model = self._copy_model()
        self.model_size_bytes = self.report_model_size()
        self.inbox = []
        self.peers = []

        self.last_cc = 0
        self.last_comm_time_s = 0.0  # Communication time derived from per-epoch communication cost.

        self.train_correct = 0
        self.train_total   = 0

    def _copy_model(self):
        return copy.deepcopy(self.model.model).to(self.device)

    def report_model_size(self):
        total_size = 0
        for param in self.neural_net_model.parameters():
            total_size += param.nelement() * param.element_size()
        print(f"Node {self.unique_id + 1} Model Size: {total_size} bytes")
        return total_size

    def set_peers(self, peers):
        self.peers = peers

    def step(self):
        import time
        
        # Compute-only timing
        compute_start = time.time()
        tic = time.time()
        result = train_simulated(
            unique_id=self.unique_id,
            model=self.neural_net_model,
            Training_ds=self.Training_ds,
            Training_lbls=self.Training_lbls,
            Testing_ds=self.Testing_ds,
            Testing_lbls=self.Testing_lbls,
            device=self.device,
            args=self.args,
            sync_callback=self.send_weights,
            epochs_override=1,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            dataset_mode=self.dataset_mode,
            array_eval=self.array_eval
        )

        compute_end = time.time()
        self.processing_time = compute_end - compute_start  # compute-only time
        
        (self.fold_idx_loss,
        self.fold_idx_accuracy,
        correct_test,
        test_size,
        processing_time,
        correct_train,
        total_train) = result

        self.correct_classifications = correct_test
        self.test_set_size           = test_size
        self.processing_time         = time.time() - tic
        # Scale compute time by capacity (uniform in [1.0, capacity_max])
        self.processing_time *= getattr(self, "compute_capacity", 1.0)

        # Store training counts for model-level aggregation.
        self.train_correct = correct_train
        self.train_total   = total_train
        
        self.correct_classifications = correct_test
        self.test_set_size = test_size

        # --- Communication timing ---
        self.merge_inbox()


    def send_weights(self, *_):
        state_dict = {k: v.cpu().clone() for k, v in self.neural_net_model.state_dict().items()}
        num_peers = len(self.peers)

        # Per-epoch communication cost for peer exchanges.
        comm_cost = num_peers * self.model_size_bytes

        # Send weights to ring neighbor(s).
        for peer in self.peers:
            peer.receive_weights_from_peer(state_dict)

        # Store communication cost and convert it to time using the uniform bandwidth assumption.
        self.last_cc = comm_cost
        bw_Bps = self.args.net_bw_mbps * 125000.0   # 100 Mbps = 12,500,000 B/s
        self.last_comm_time_s = self.last_cc / bw_Bps

        print(
            f"Communication cost for Node {self.unique_id + 1}: {comm_cost} bytes "
            f"(~{self.last_comm_time_s:.3f} s at {self.args.net_bw_mbps} Mbps)"
        )
        return comm_cost


    def receive_weights_from_peer(self, peer_weights):
        self.inbox.append(peer_weights)

    def merge_inbox(self):
        if not self.inbox:
            return

        device = self.device
        avg_state_dict = {}
        local_state_dict = self.neural_net_model.state_dict()

        # Average local model + received peer model(s)
        num_models = len(self.inbox) + 1

        for k, v in local_state_dict.items():
            if v.dtype in (torch.float32, torch.float64, torch.float16):
                avg = v.to(device).float().clone()

                for weights in self.inbox:
                    avg += weights[k].to(device).float()

                avg /= num_models
                avg_state_dict[k] = avg.to(v.dtype)
            else:
                # Keep local non-floating buffers unchanged
                avg_state_dict[k] = v.to(device)

        self.neural_net_model.load_state_dict(avg_state_dict)
        self.inbox.clear()
  