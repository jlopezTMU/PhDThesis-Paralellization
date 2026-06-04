"""
DLMP: Deep Learning Multi-Processing simulator

Author: Jorge A. Lopez
Affiliation: Toronto Metropolitan University

Description:
Agent-based simulation framework for studying coordination strategies
in distributed deep learning systems.

Repository:
https://github.com/DLMPsim/DLMP

License:
MIT License
"""

import copy
import time
from mesa import Agent
from trainMASCNN import train_simulated, build_loaders
import torch


class ProcessorAgent(Agent):
    """
    Simulated processor (node) that trains on its partition and then
    participates in global synchronous weight averaging.
    """
    def __init__(
        self, unique_id, model,
        Training_ds, Training_lbls,
        Testing_ds,  Testing_lbls,
        device, args
    ):
        super().__init__(unique_id, model)
        self.Training_ds   = Training_ds
        self.Training_lbls = Training_lbls
        self.Testing_ds    = Testing_ds
        self.Testing_lbls  = Testing_lbls
        self.device        = device
        self.args          = args
        
        # Build DataLoaders once per agent to avoid recreating them each epoch.
        self.train_loader, self.test_loader, self.dataset_mode, self.array_eval = build_loaders(
            self.Training_ds, self.Training_lbls,
            self.Testing_ds,  self.Testing_lbls,
            self.device, self.args
        )


        self.compute_capacity = 1.0   # set by the Model after agent creation; minimum capacity is 1.0
        self.processing_time = 0.0    # Initialize to numeric value to avoid None-related errors


        self.neural_net_model = self._copy_model().to(device)
        self.report_model_size()

        # Metrics set after every global epoch
        
        self.fold_idx_loss          = None
        self.fold_idx_accuracy      = None
        self.correct_classifications = 0
        # In dataset-mode, Testing_lbls is a list of indices (agent shard)
        self.test_set_size = len(self.Testing_lbls) if self.Testing_lbls is not None else len(self.Testing_ds)

        # For training-accuracy aggregation (all datasets)
        self.train_correct          = 0
        self.train_total            = 0
        # Communication metrics collected each global epoch.
        self.last_cc = 0
        self.last_comm_time_s = 0.0

    # ---------------------------------------------------------------------
    #  Helper utilities
    # --------------------------------------------------------------------- 
    def _copy_model(self):
        return copy.deepcopy(self.model.model)

    def report_model_size(self):
        tot = sum(p.nelement() * p.element_size()
                  for p in self.neural_net_model.parameters())
        self.model_size_bytes = tot         
        print(f"Node {self.unique_id + 1} Model Size: {tot} bytes")

    # --------------------------------------------------------------------- 
    #  Core Mesa step – one GLOBAL epoch
    # --------------------------------------------------------------------- 
    def step(self):
        tic = time.time()

        # ---------------------------------------------------------------
        #  Run exactly one local epoch during each global epoch.
        # ---------------------------------------------------------------
        orig_epochs      = self.args.epochs   
        self.args.epochs = 1                  

        result = train_simulated(
            unique_id      = self.unique_id,
            model          = self.neural_net_model,
            Training_ds    = self.Training_ds,
            Training_lbls  = self.Training_lbls,
            Testing_ds     = self.Testing_ds,
            Testing_lbls   = self.Testing_lbls,
            device         = self.device,
            args           = self.args,
            sync_callback  = (lambda _weights: None),
            train_loader   = self.train_loader,
            test_loader    = self.test_loader,
            dataset_mode   = self.dataset_mode,
            array_eval     = self.array_eval
        )


        self.args.epochs = orig_epochs        

        (self.fold_idx_loss,
         self.fold_idx_accuracy,
         correct_test,
         test_size,
         _,
         correct_train,
         total_train) = result

        # Keep existing test metrics
        self.correct_classifications = correct_test
        self.test_set_size           = test_size
        self.processing_time         = time.time() - tic

        # Scale compute time by capacity (uniform in [1.0, capacity_max])
        self.processing_time *= getattr(self, "compute_capacity", 1.0)


        # Store training counts for model-level aggregation
        self.train_correct = correct_train
        self.train_total   = total_train


        # ---------------------------------------------------------------
        #  Print per-node communication cost once this global epoch
        # ---------------------------------------------------------------

        num_agents = self.model.num_processors
        # Approximate per-node communication cost per epoch:
        # (n - 1) * model_size_bytes * 2 (send + receive)
        self.last_cc = (num_agents - 1) * self.model_size_bytes * 2

        bw_Bps = self.args.net_bw_mbps * 125000.0  # 100 Mbps -> 12,500,000 B/s
        self.last_comm_time_s = self.last_cc / bw_Bps

        print(f"Node {self.unique_id + 1} Model Communication Cost: {self.last_cc} bytes "
        f"(~{self.last_comm_time_s:.3f} s at {self.args.net_bw_mbps} Mbps)")
    # --------------------------------------------------------------------- #
    #  Called by model after averaging weights
    # --------------------------------------------------------------------- #
    def set_weights_from_averaged_state_dict(self, avg_state_dict):
        self.neural_net_model.load_state_dict({
            k: avg_state_dict[k].to(v.dtype)
            for k, v in self.neural_net_model.state_dict().items()
        })

