from mesa import Agent
import time
from trainMASTransformers import train_simulated

class ProcessorAgent(Agent):
    def __init__(self, unique_id, model, train_idx, val_idx, X, y, device, args):
        super().__init__(unique_id, model)
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.X = X
        self.y = y
        self.device = device
        self.args = args
        self.fold_loss = None
        self.fold_accuracy = None
        self.model = None
        self.processing_time = None
        self.parallel_model = model

    def step(self):
        start_time = time.time()
        self.fold_loss, self.fold_accuracy, self.model = train_simulated(
            self.unique_id, self.train_idx, self.val_idx, 
            self.X, self.y, self.device, self.args
        )
        end_time = time.time()
        self.processing_time = end_time - start_time

        print(f"--- Node {self.unique_id + 1}/{self.parallel_model.num_processors} completed in {self.processing_time:.4f} seconds ---")
        print(f"--- Node {self.unique_id + 1} Accuracy: {self.fold_accuracy * 100:.2f}% ---")
