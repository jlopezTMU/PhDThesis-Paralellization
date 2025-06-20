import copy
from mesa import Agent
from trainMASACNN import train_simulated
import torch

class ProcessorAgent(Agent):
    """
    Represents a simulated processor (or node) that trains on a subset of the data.
    Each agent trains its local copy of a neural network and participates in peer-to-peer
    asynchronous weight exchange.
    """
    def __init__(self, unique_id, model, Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args):
        super().__init__(unique_id, model)
        self.Training_ds = Training_ds
        self.Training_lbls = Training_lbls
        self.Testing_ds = Testing_ds
        self.Testing_lbls = Testing_lbls
        self.device = device
        self.args = args

        self.fold_idx_loss = None
        self.fold_idx_accuracy = None
        self.processing_time = None
        self.correct_classifications = 0
        self.test_set_size = len(self.Testing_ds)

        self.neural_net_model = self._copy_model()
        self.model_size_bytes = self.report_model_size()   # --- CC CHANGE ---
        self.inbox = []  # Inbox for receiving peer weights
        self.peers = []  # List of peer agents

        self.last_cc = 0   # --- CC CHANGE ---

    def _copy_model(self):
        return copy.deepcopy(self.model.model).to(self.device)

    def report_model_size(self):
        total_size = 0
        for param in self.neural_net_model.parameters():
            total_size += param.nelement() * param.element_size()
        print(f"Node {self.unique_id + 1} Model Size: {total_size} bytes")
        return total_size   # --- CC CHANGE ---

    def set_peers(self, peers):
        self.peers = peers

    def step(self):
        """
        Executes one training step using the model, training, and testing data.
        Then performs peer-to-peer communication: sends weights and merges inbox.
        """
        import time
        start_time = time.time()

        result = train_simulated(
            unique_id=self.unique_id,
            model=self.neural_net_model,
            Training_ds=self.Training_ds,
            Training_lbls=self.Training_lbls,
            Testing_ds=self.Testing_ds,
            Testing_lbls=self.Testing_lbls,
            device=self.device,
            args=self.args,
            sync_callback=lambda _: None  # No global sync
        )

        self.fold_idx_loss, self.fold_idx_accuracy, correct_test, test_size, proc_time = result
        end_time = time.time()
        self.processing_time = end_time - start_time

        self.correct_classifications = correct_test
        self.test_set_size = test_size

        self.last_cc = self.send_weights()  # --- CC CHANGE ---
        self.merge_inbox()

    def send_weights(self):
        state_dict = {k: v.cpu().clone() for k, v in self.neural_net_model.state_dict().items()}
        num_peers = len(self.peers)
        # CC = (#peers) × (model_size) × 2 (send + receive, as per Jorge's definition)
        comm_cost = num_peers * self.model_size_bytes * 2   # --- CC CHANGE ---
        for peer in self.peers:
            peer.receive_weights_from_peer(state_dict)
        print(f"Communication cost for Node {self.unique_id + 1}: {comm_cost} bytes")  # --- CC CHANGE ---
        return comm_cost   # --- CC CHANGE ---

    def receive_weights_from_peer(self, peer_weights):
        self.inbox.append(peer_weights)

    def merge_inbox(self):
        if not self.inbox:
            return
        num_models = len(self.inbox)
        avg_state_dict = {k: torch.zeros_like(v) for k, v in self.neural_net_model.state_dict().items()}

        for weights in self.inbox:
            for k in weights:
                avg_state_dict[k] += weights[k].to(self.device)

        for k in avg_state_dict:
            avg_state_dict[k] /= num_models

        self.neural_net_model.load_state_dict(avg_state_dict)
        self.inbox.clear()
