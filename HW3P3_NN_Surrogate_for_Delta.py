import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import HW3_P2_old

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class OptionPricingNN(nn.Module):
    """Neural Network for option pricing with single hidden layer"""

    def __init__(self, input_dim=1, hidden_dim=30, output_dim=1, activation='relu'):
        super(OptionPricingNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Choose activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError("Activation must be 'relu', 'tanh', or 'elu'")

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepOptionPricingNN(nn.Module):
    """Neural Network with multiple hidden layers"""

    def __init__(self, input_dim=1, hidden_dims=[30, 20, 10], output_dim=1, activation='relu'):
        super(DeepOptionPricingNN, self).__init__()

        # Choose activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError("Activation must be 'relu', 'tanh', or 'elu'")

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_neural_network(X_train, y_train, X_val, y_val, model, epochs=10000, lr=0.001, patience=50):
    """Train neural network with early stopping"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # Load best model
    model.load_state_dict(best_model_state)
    return train_losses, val_losses


def compute_nn_delta(model, scaler, S0_values):
    """Compute Delta using automatic differentiation"""
    deltas = []

    for S0 in S0_values:
        # Convert to tensor and enable gradient computation
        S0_tensor = torch.tensor([[S0]], dtype=torch.float32, requires_grad=True)

        # Normalize input
        S0_normalized = torch.tensor(scaler.transform([[S0]]), dtype=torch.float32, requires_grad=True)

        # Forward pass
        price = model(S0_normalized)

        # Compute gradient
        price.backward()
        delta = S0_normalized.grad.item() / scaler.scale_[0]  # Adjust for normalization
        deltas.append(delta)

        # Clear gradients
        S0_normalized.grad.zero_()

    return np.array(deltas)


# Generate training data using Problem 2 functions
print("Generating training data...")
S0_range = np.linspace(8, 12, 1000)        # 1000 evenly-spaced inputs
# or:  S0_range = 8 + 4*np.random.rand(1000)   # 1000 i.i.d. uniform samples

M = 1000

# Generate option prices using Monte Carlo from Problem 2
prices = []
for i, S0 in enumerate(S0_range):
    if i % 100 == 0:
        print(f"Processing S0 = {S0:.1f}")

    seed_offset = i * 10000
    price = HW3_P2.price_call_mc(S0, M=M, seed_offset=seed_offset)
    prices.append(price)

prices = np.array(prices)

# Prepare data for neural network
X = S0_range.reshape(-1, 1)
y = prices.reshape(-1, 1)

# Split into train/validation sets
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Normalize features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val)

# Test different activation functions
activations = ['relu', 'tanh', 'elu']
models = {}
results = {}

print("\nTraining neural networks with different activation functions...")

for activation in activations:
    print(f"\nTraining model with {activation.upper()} activation...")

    # Create and train model
    model = OptionPricingNN(input_dim=1, hidden_dim=30, output_dim=1, activation=activation)
    train_losses, val_losses = train_neural_network(
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
        model, epochs=10000, lr=0.001, patience=50
    )

    models[activation] = model
    results[activation] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_val_loss': val_losses[-1]
    }

# Compute predictions and deltas for all models
print("\nComputing predictions and deltas...")
X_full_scaled = scaler_X.transform(X)
X_full_tensor = torch.FloatTensor(X_full_scaled)

predictions = {}
deltas_nn = {}

for activation in activations:
    model = models[activation]
    model.eval()

    with torch.no_grad():
        pred = model(X_full_tensor).numpy().flatten()

    predictions[activation] = pred
    deltas_nn[activation] = compute_nn_delta(model, scaler_X, S0_range)

# Compute reference deltas from Problem 2
print("\nComputing reference deltas using bump-and-revalue...")
deltas_bump = []
for i, S0 in enumerate(S0_range):
    if i % 10 == 0:
        print(f"Processing S0 = {S0:.1f}")

    seed_offset = i * 10000
    delta = HW3_P2.compute_delta_bump_revalue(S0, M=1000, seed_offset=seed_offset)
    deltas_bump.append(delta)

deltas_bump = np.array(deltas_bump)

# Try deeper architecture
print("\nTraining deeper neural network...")
deep_model = DeepOptionPricingNN(input_dim=1, hidden_dims=[30, 20, 10], output_dim=1, activation='relu')
train_losses_deep, val_losses_deep = train_neural_network(
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
    deep_model, epochs=10000, lr=0.001, patience=50
)

with torch.no_grad():
    predictions_deep = deep_model(X_full_tensor).numpy().flatten()

deltas_deep = compute_nn_delta(deep_model, scaler_X, S0_range)

# Plotting results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Option prices comparison
axes[0, 0].plot(S0_range, prices, 'ko', label='MC Prices', markersize=4)
for activation in activations:
    axes[0, 0].plot(S0_range, predictions[activation], '-', label=f'NN ({activation.upper()})', linewidth=2)
axes[0, 0].plot(S0_range, predictions_deep, '--', label='Deep NN', linewidth=2)
axes[0, 0].set_xlabel('Initial Stock Price S₀')
axes[0, 0].set_ylabel('Option Price')
axes[0, 0].set_title('Neural Network Predictions vs Monte Carlo')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training curves
for activation in activations:
    axes[0, 1].plot(results[activation]['train_losses'], label=f'Train ({activation.upper()})')
    axes[0, 1].plot(results[activation]['val_losses'], '--', label=f'Val ({activation.upper()})')
axes[0, 1].plot(train_losses_deep, label='Train (Deep)', linewidth=2)
axes[0, 1].plot(val_losses_deep, '--', label='Val (Deep)', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Training Curves')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# Plot 3: Delta comparison
axes[0, 2].plot(S0_range, deltas_bump, 'ko-', label='Bump-and-Revalue', markersize=3)
for activation in activations:
    axes[0, 2].plot(S0_range, deltas_nn[activation], '-', label=f'NN ({activation.upper()})', linewidth=2)
axes[0, 2].plot(S0_range, deltas_deep, '--', label='Deep NN', linewidth=2)
axes[0, 2].set_xlabel('Initial Stock Price S₀')
axes[0, 2].set_ylabel('Delta')
axes[0, 2].set_title('Delta Comparison')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Price prediction errors
for activation in activations:
    axes[1, 0].plot(S0_range, np.abs(prices - predictions[activation]),
                    label=f'NN ({activation.upper()})', linewidth=2)
axes[1, 0].plot(S0_range, np.abs(prices - predictions_deep), '--',
                label='Deep NN', linewidth=2)
axes[1, 0].set_xlabel('Initial Stock Price S₀')
axes[1, 0].set_ylabel('Absolute Price Error')
axes[1, 0].set_title('Price Prediction Errors')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# Plot 5: Delta errors
for activation in activations:
    axes[1, 1].plot(S0_range, np.abs(deltas_bump - deltas_nn[activation]),
                    label=f'NN ({activation.upper()})', linewidth=2)
axes[1, 1].plot(S0_range, np.abs(deltas_bump - deltas_deep), '--',
                label='Deep NN', linewidth=2)
axes[1, 1].set_xlabel('Initial Stock Price S₀')
axes[1, 1].set_ylabel('Absolute Delta Error')
axes[1, 1].set_title('Delta Prediction Errors')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Final validation losses comparison
val_losses_final = [results[act]['final_val_loss'] for act in activations]
val_losses_final.append(val_losses_deep[-1])
activation_labels = [act.upper() for act in activations] + ['DEEP']
axes[1, 2].bar(activation_labels, val_losses_final, alpha=0.7)
axes[1, 2].set_ylabel('Final Validation Loss')
axes[1, 2].set_title('Model Comparison')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "=" * 60)
print("NEURAL NETWORK SURROGATE RESULTS SUMMARY")
print("=" * 60)

print("\nFinal Validation Losses:")
for activation in activations:
    print(f"{activation.upper()}: {results[activation]['final_val_loss']:.6f}")
print(f"DEEP: {val_losses_deep[-1]:.6f}")

print("\nMean Absolute Price Errors:")
for activation in activations:
    mae = np.mean(np.abs(prices - predictions[activation]))
    print(f"{activation.upper()}: {mae:.6f}")
mae_deep = np.mean(np.abs(prices - predictions_deep))
print(f"DEEP: {mae_deep:.6f}")

print("\nMean Absolute Delta Errors:")
for activation in activations:
    mae_delta = np.mean(np.abs(deltas_bump - deltas_nn[activation]))
    print(f"{activation.upper()}: {mae_delta:.6f}")
mae_delta_deep = np.mean(np.abs(deltas_bump - deltas_deep))
print(f"DEEP: {mae_delta_deep:.6f}")

print("\nMax Absolute Delta Errors:")
for activation in activations:
    max_delta = np.max(np.abs(deltas_bump - deltas_nn[activation]))
    print(f"{activation.upper()}: {max_delta:.6f}")
max_delta_deep = np.max(np.abs(deltas_bump - deltas_deep))
print(f"DEEP: {max_delta_deep:.6f}")

# Determine best performing model
best_activation = min(activations, key=lambda x: results[x]['final_val_loss'])
print(f"\nBest performing single-layer activation: {best_activation.upper()}")

if val_losses_deep[-1] < results[best_activation]['final_val_loss']:
    print("Deep architecture outperforms single-layer models")
else:
    print("Single-layer architecture is sufficient")

print("\nAnalysis:")
print("- ReLU: Generally stable, good for deep networks, but can suffer from dead neurons")
print("- Tanh: Bounded output, good gradient flow, but can saturate")
print("- ELU: Smooth everywhere, can help with vanishing gradients")
print("- Deep networks: More parameters but risk of overfitting with limited data")