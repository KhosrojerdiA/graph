 
import torch

from utils.utils import create_training_set_for_trained_model
import torch.nn as nn
import torch.optim as optim

#____________________________________________________________________________________________________________________________________________


def train_rl_sv2(training_set, input_dim, hidden_dim, lr):


    epochs = 1000

        # RL Agent Definition
    class RLAgent(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(RLAgent, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)  # Output a score for edge removal
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x).squeeze()
    # Initialize RL Agent
    model = RLAgent(input_dim, hidden_dim).to(training_set.x.device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr)
    loss_fn = nn.MSELoss()

    # Generate feature matrix for the training set
    features = training_set.x  # Assuming node features exist
    edge_index = training_set.edge_index  # Training edge index

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        edge_scores = model(features)  # Get scores for all nodes
        
        # Use scores to predict edge importance
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        edge_predictions = (edge_scores[src_nodes] + edge_scores[dst_nodes]) / 2  # Averaging node scores

        # Generate target labels (dummy labels for now)
        target_labels = torch.ones_like(edge_predictions)

        # Compute loss
        loss = loss_fn(edge_predictions, target_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


#____________________________________________________________________________________________________________________________________________

def create_train_rl_sv2_model(data, selected_nodes):

    training_set = create_training_set_for_trained_model(data, selected_nodes)
    print("_____________________________________")
    print("Started Training rl model!")
    print("_____________________________________")
    input_dim = data.num_features
    hidden_dim = 64
    lr = 0.01
    rl_model = train_rl_sv2(training_set, input_dim, hidden_dim, lr)
    print(f"rl model is trained!")
    
    return rl_model

#____________________________________________________________________________________________________________________________________________
