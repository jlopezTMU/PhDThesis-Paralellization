import copy
from mesa import Agent
from trainMASCNN import train_simulated

class ProcessorAgent(Agent):
    """
    Represents a simulated processor (or node) that trains on a subset of the data.
    Each agent trains its local copy of a neural network and participates in global
    weight synchronization.
    """
    def __init__(self, unique_id, model, Training_ds, Training_lbls, Testing_ds, Testing_lbls, device, args):
        # 'model' here is the simulation model (i.e., ParallelizationModel)
        super().__init__(unique_id, model)
        self.Training_ds = Training_ds
        self.Training_lbls = Training_lbls
        self.Testing_ds = Testing_ds
        self.Testing_lbls = Testing_lbls
        self.device = device
        self.args = args

        # Metrics for training
        self.fold_idx_loss = None
        self.fold_idx_accuracy = None
        self.processing_time = None
        self.correct_classifications = 0
        self.test_set_size = len(self.Testing_ds)

        # Create a deep copy of the shared neural network model from the simulation model.
        self.neural_net_model = self._copy_model()
        self.report_model_size()

    def _copy_model(self):
        """Creates a deep copy of the shared neural network model."""
        # 'self.model' is the simulation model; its attribute 'model' holds the shared NN
        return copy.deepcopy(self.model.model).to(self.device)

    def report_model_size(self):
        """Calculates and prints the size of the local neural network model (in bytes)."""
        total_size = 0
        for param in self.neural_net_model.parameters():
            total_size += param.nelement() * param.element_size()
        print(f"Node {self.unique_id + 1} Model Size: {total_size} bytes")

    def step(self):
        """
        Executes one training step using the model, training, and testing data.
        Calls the training simulation and (in this sync version) does NOT
        synchronize weights from within the agent—this is handled globally by the model.
        """
        import time
        start_time = time.time()

        def sync_callback(model_state_dict):
            # No-op: Synchronization is handled by the model after all agents finish.
            pass

        result = train_simulated(
            unique_id=self.unique_id,
            model=self.neural_net_model,
            Training_ds=self.Training_ds,
            Training_lbls=self.Training_lbls,
            Testing_ds=self.Testing_ds,
            Testing_lbls=self.Testing_lbls,
            device=self.device,
            args=self.args,
            sync_callback=sync_callback  # <-- No-op callback
        )

        self.fold_idx_loss, self.fold_idx_accuracy, correct_test, test_size, proc_time = result
        end_time = time.time()
        self.processing_time = end_time - start_time

        self.correct_classifications = correct_test
        self.test_set_size = test_size
