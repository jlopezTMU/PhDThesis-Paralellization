from mesa import Model
from mesa.time import SimultaneousActivation
from sklearn.model_selection import KFold
from MASAgentTransformers import ProcessorAgent

class ParallelizationModel(Model):
    def __init__(self, X, y, device, args):
        super().__init__()
        self.num_processors = args.processors
        self.X = X
        self.y = y
        self.device = device
        self.args = args
        self.schedule = SimultaneousActivation(self)

        kf = KFold(n_splits=self.num_processors)
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            agent = ProcessorAgent(i, self, train_idx, val_idx, X, y, device, args)
            self.schedule.add(agent)

    def step(self):
        self.schedule.step()
        losses = [agent.fold_loss for agent in self.schedule.agents]
        accuracies = [agent.fold_accuracy for agent in self.schedule.agents]
        processing_times = [agent.processing_time for agent in self.schedule.agents]

        average_loss = sum(losses) / len(losses)
        average_accuracy = sum(accuracies) / len(accuracies)
        cumulative_time = sum(processing_times)
        average_time_per_node = cumulative_time / len(processing_times)

        print(f"--- Cumulative processing time for all nodes: {cumulative_time:.4f} seconds ---")
        print(f"--- Average processing time per node: {average_time_per_node:.4f} seconds ---")
        print(f"Average Loss: {average_loss:.4f}, Average Accuracy: {average_accuracy * 100:.2f}%")
