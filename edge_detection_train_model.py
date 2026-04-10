import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ==========================================
# 1. DEFINE THE NEURAL NETWORK
# ==========================================

class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Two layers of graph convolution (message passing)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Pass features through layer 1, apply ReLU activation
        x = self.conv1(x, edge_index).relu()
        # Pass through layer 2
        return self.conv2(x, edge_index)

class EdgeDecoder(torch.nn.Module):
    def forward(self, z, edge_label_index):
        # z contains the "learned profiles" (embeddings) for every person
        # We grab the profiles for the specific Senders (row 0) and Receivers (row 1) we are testing
        senders = z[edge_label_index[0]]
        receivers = z[edge_label_index[1]]
        
        # Multiply them together and sum the result to get a single score per pair
        return (senders * receivers).sum(dim=-1)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = EdgeDecoder()

    def forward(self, x, edge_index, edge_label_index):
        # Step 1: Encode the whole graph
        z = self.encoder(x, edge_index)
        # Step 2: Decode the specific edges we want to predict
        return self.decoder(z, edge_label_index)


# ==========================================
# 2. LOAD DATA & PERFORM TRAINING ON 80% DATA SET
# ==========================================

if __name__ == "__main__":
    print("1. Loading saved graph data...")
    dataset = torch.load('enron_graph_data.pt', weights_only=False)
    train_data = dataset['train']
    test_data = dataset['test']
    
    # Convert integer features to floats for the neural network
    x_float = train_data.x.float()

    # Compress the massive email counts using a log transform
    x_float = torch.log1p(x_float)
    
    print("2. Initializing Model and Optimizer...")
    # Initialize the model (2 input features: In-Degree and Out-Degree)
    model = LinkPredictor(in_channels=2, hidden_channels=16, out_channels=8)
    
    # The Optimizer handles updating the weights. Learning rate (lr) controls how fast it learns.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # The Loss Function (BCE) is perfect for Yes/No (1 or 0) questions like Link Prediction.
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- THE TRAINING LOOP ---
    def train():
        model.train()           # Put the model in training mode
        optimizer.zero_grad()   # Clear out the old math from the previous step
        
        # 1. Forward Pass: Make a guess for every edge in the training set
        predictions = model(x_float, train_data.edge_index, train_data.edge_label_index)
        
        # 2. Calculate the Loss: Compare guesses to the real answers (edge_label)
        loss = criterion(predictions, train_data.edge_label.float())
        
        # 3. Backward Pass: Figure out which weights need to change
        loss.backward()
        
        # 4. Update the Weights: Actually apply the changes
        optimizer.step()
        
        return loss.item()

    print("\n3. Starting Training...")
    epochs = 100 # How many times we read the whole dataset
    
    for epoch in range(1, epochs + 1):
        loss = train()
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

    print("\nTraining Complete! Your model has learned the Enron structure.")

# ==========================================
# 4. EVALUATION OF THE MODEL ON THE TEST SET
# ==========================================

print("\n4. Running the Final Exam (Evaluation)...")

# @torch.no_grad() is a command that tells PyTorch: 
# "Stop tracking math for training, we are just taking a test now."
# It saves memory and makes the code run instantly.
@torch.no_grad()
def evaluate(data):
    model.eval()  # Put the model into 'Testing Mode'
    
    # 1. Apply the exact same log normalization to the test features
    x_test_float = torch.log1p(data.x.float())
    
    # 2. Make predictions on the hidden edges
    raw_scores = model(x_test_float, data.edge_index, data.edge_label_index)
    
    # 3. Convert raw scores to Yes/No (1 or 0)
    # Because we used BCEWithLogitsLoss, a score > 0 means the model 
    # thinks the probability is greater than 50% that an email exists.
    predictions = (raw_scores > 0).float()
    
    # 4. Grade the test: Compare predictions to the actual answer key
    correct_guesses = (predictions == data.edge_label).sum().item()
    total_questions = data.edge_label.size(0)
    
    accuracy = correct_guesses / total_questions
    return accuracy

# Run the evaluation on our hidden test set
test_accuracy = evaluate(test_data)

print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")