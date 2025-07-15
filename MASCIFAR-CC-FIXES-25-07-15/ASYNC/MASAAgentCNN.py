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
        self.model_size_bytes = self.report_model_size()
        self.inbox = []  # Inbox for receiving peer weights
        self.peers = []  # List of peer agents

        self.last_cc = 0  # Will be set after send_weights()

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

        self.last_cc = self.send_weights()
        self.merge_inbox()

    def send_weights(self):
        """
        Sends this agent's weights to all peers and calculates communication cost.
        Dense async: each agent sends to (n - 1) peers and receives from (n - 1).
        Total per-agent CC = 2 * (n - 1) * |W|
        """
        state_dict = {k: v.cpu().clone() for k, v in self.neural_net_model.state_dict().items()}
        num_peers = len(self.peers)

        # Sanity check
        assert num_peers == self.args.processors - 1, (
            f"[ERROR] Node {self.unique_id + 1}: expected {self.args.processors - 1} peers, got {num_peers}"
        )

        # Communication cost: (n - 1 sends + n - 1 receives) × |W|
        comm_cost = num_peers * self.model_size_bytes * 2

        for peer in self.peers:
            peer.receive_weights_from_peer(state_dict)

        print(f"Communication cost for Node {self.unique_id + 1}: {comm_cost} bytes")
        print(f"[DEBUG] Node {self.unique_id + 1} sent weights to {num_peers} peers")
        return comm_cost

    def receive_weights_from_peer(self, peer_weights):
        self.inbox.append(peer_weights)

    def merge_inbox(self):
        if not self.inbox:
            return
        num_models = len(self.inbox)
        ref_state = self.neural_net_model.state_dict()
        avg_state_dict = {}

        # Initialize accumulator with proper types
        for k, v in ref_state.items():
            if torch.is_floating_point(v):
                avg_state_dict[k] = torch.zeros_like(v)
            else:
                avg_state_dict[k] = v.clone()  # do not average ints

        for weights in self.inbox:
            for k in weights:
                v = weights[k].to(self.device)
                if torch.is_floating_point(v):
                    avg_state_dict[k] += v
                else:
                    avg_state_dict[k] = v  # keep as-is

        for k in avg_state_dict:
            if torch.is_floating_point(avg_state_dict[k]):
                avg_state_dict[k] /= float(num_models)

        self.neural_net_model.load_state_dict(avg_state_dict)
        self.inbox.clear()
