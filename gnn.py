import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class GNNConfig:
    def __init__(self):
        self.NUM_CLASSES = 40  # Same as SlowFast activity classes
        self.NUM_FEATURES = 6  # Features: [x_center, y_center, width, height, speed_x, speed_y]
        self.HIDDEN_DIM = 32  # Reduced for CPU
        self.NUM_EPOCHS = 30  # Reduced for faster training on CPU
        self.BATCH_SIZE = 16  # Reduced for CPU
        self.LEARNING_RATE = 0.005  # Learning rate
        self.DEVICE = torch.device('cpu')  # Explicitly set to CPU
        self.GRAPH_FILE = "relationship_graphs.pt"  # Path to precomputed graphs
        # Paths to video folders (for reference, not used directly here)
        self.TRACKED_VIDEOS_DIR = "tracked_videos"
        self.SLOWFAST_VIDEOS_DIR = "slowfast_activity_videos"
        # Activity classes (same as SlowFast and create_relationship_graphs.py)
        self.ACTIVITY_CLASSES = [
            'walking', 'running', 'jogging', 'jumping', 'sitting', 'standing',
            'waving', 'clapping', 'dancing', 'eating', 'drinking', 'talking',
            'reading', 'writing', 'sleeping', 'exercising', 'playing_sports',
            'cooking', 'cleaning', 'driving', 'cycling', 'swimming', 'climbing',
            'carrying', 'pushing', 'pulling', 'throwing', 'catching', 'kicking',
            'punching', 'hugging', 'shaking_hands', 'applauding', 'crawling',
            'falling_down', 'getting_up', 'lying_down', 'bending', 'stretching'
        ]

# GNN Model Definition
class GCNActivityClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCNActivityClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Final classification layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Load precomputed relationship graphs
def load_graph_data(config):
    """
    Load relationship graphs from file, inspect them, and add self-loops if needed.
    """
    if not os.path.exists(config.GRAPH_FILE):
        logger.error(f"Graph file {config.GRAPH_FILE} does not exist. Run create_relationship_graphs.py first.")
        return []
    
    try:
        # Set weights_only=False (can change to True if file is trusted)
        data_list = torch.load(config.GRAPH_FILE, weights_only=False)
        logger.info(f"Loaded {len(data_list)} graphs from {config.GRAPH_FILE}")
        
        # Inspect and fix graphs
        valid_data_list = []
        for idx, data in enumerate(data_list):
            num_nodes = data.x.shape[0]
            num_edges = data.edge_index.shape[1] if data.edge_index.numel() > 0 else 0
            logger.info(f"Graph {idx}: {num_nodes} nodes, {num_edges} edges")
            
            # Validate data
            if data.x.shape[1] != config.NUM_FEATURES:
                logger.error(f"Feature dimension mismatch in graph {idx}: expected {config.NUM_FEATURES}, got {data.x.shape[1]}")
                return []
            if not (0 <= data.y.min() <= data.y.max() < config.NUM_CLASSES):
                logger.error(f"Invalid labels in graph {idx}: {data.y}")
                return []
            
            # Add self-loops if the graph has no edges
            if num_edges == 0:
                logger.warning(f"Graph {idx} has no edges. Adding self-loops to prevent GCNConv failure.")
                edge_index, _ = add_self_loops(data.edge_index, num_nodes=num_nodes)
                data.edge_index = edge_index
                logger.info(f"Graph {idx} after adding self-loops: {data.edge_index.shape[1]} edges")
            
            valid_data_list.append(data)
        
        return valid_data_list
    except Exception as e:
        logger.error(f"Failed to load graphs: {e}")
        return []

# Training and Testing
class GNNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.model = GCNActivityClassifier(
            num_features=config.NUM_FEATURES,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_CLASSES
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE) # optimizer
        self.criterion = torch.nn.NLLLoss()

    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def test(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = self.model(data)
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        # Get unique labels present in the data
        unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
        # Map the unique labels to their corresponding activity names
        target_names = [self.config.ACTIVITY_CLASSES[i] for i in unique_labels]
        
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names, zero_division=0)
        return accuracy, report

def main():
    # Configuration
    config = GNNConfig()
    logger.info(f"Using device: {config.DEVICE}")

    # Load dataset
    logger.info("Loading graph dataset...")
    data_list = load_graph_data(config)
    if not data_list:
        logger.error("No graphs loaded. Exiting...")
        return

    # Split into train and test (80-20 split)
    train_data = data_list[:int(0.8 * len(data_list))]
    test_data = data_list[int(0.8 * len(data_list)):]
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)
    logger.info(f"Dataset prepared: {len(train_data)} training graphs, {len(test_data)} testing graphs")

    # Initialize trainer
    trainer = GNNTrainer(config)
    logger.info("Starting GNN training on CPU...")

    # Training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        loss = trainer.train(train_loader)
        if epoch % 5 == 0:  # Log more frequently since fewer epochs
            logger.info(f"Epoch {epoch}/{config.NUM_EPOCHS}, Loss: {loss:.4f}")

    # Testing
    logger.info("Testing GNN model on CPU...")
    accuracy, report = trainer.test(test_loader)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    logger.info("\n" + report)

    # Save model
    model_path = "gnn_model_cpu.pth"
    torch.save(trainer.model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()