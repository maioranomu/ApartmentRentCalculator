import torch
import torch.nn as nn
import torch.optim as optim
import os
name = "Milene Zanardi Maioranov"
"""
Region:
1 : 6x
2 : 3x
3 : 1x
"""
df = [
    [30, 1, 180],
    [30, 2, 90],
    [30, 3, 30],
    [40, 1, 240],
    [40, 2, 120],
    [40, 3, 40],
    [50, 1, 300],
    [50, 2, 150],
    [50, 3, 50],
    [60, 1, 360],
    [60, 2, 180],
    [60, 3, 60],
    [70, 1, 420],
    [70, 2, 210],
    [70, 3, 70],
    [80, 1, 480],
    [80, 2, 240],
    [80, 3, 80],
    [90, 1, 540],
    [90, 2, 270],
    [90, 3, 90],
    [100, 1, 600],
    [100, 2, 300],
    [100, 3, 100],
    [110, 1, 660],
    [110, 2, 330],
    [110, 3, 110],
    [120, 1, 720],
    [120, 2, 360],
    [120, 3, 120],
    [130, 1, 780],
    [130, 2, 390],
    [130, 3, 130],
    [140, 1, 840],
    [140, 2, 420],
    [140, 3, 140],
]

x = torch.tensor([[i[0], i[1]] for i in df], dtype=torch.float32)  # Input: m² and region
y = torch.tensor([[i[2]] for i in df], dtype=torch.float32)          # Target: price

# Training loop
loss_fn = nn.MSELoss()
model = nn.Sequential(
    nn.Linear(2, 25000),
    nn.ReLU(),
    nn.Linear(25000, 1),
)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model_path = os.path.join("Memory/model.pth")
optimizer_path = os.path.join("Memory/optimizer.pth")
try:
    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optimizer_path))
except:
    epochs = 50000
    for epoch in range(epochs):
        predictions = model(x)              # Forward pass
        loss = loss_fn(predictions, y)      # Compute loss
        optimizer.zero_grad()                # Reset gradients
        loss.backward()                      # Set Backward
        optimizer.step()                     # Update weights

        # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {loss.item():.4f}")
            if loss.item() <= 0.0001:
                break
    predictions = model(x)
    final_loss = loss_fn(predictions, y)
    print(f"\nFinal Loss after training: {final_loss.item():.4f}")

    torch.save(model.state_dict(), model_path)  # Salva o estado do modelo
    torch.save(optimizer.state_dict(), optimizer_path)  # Salva o estado do otimizador
    print("Saved")

#------------------------------------------
tx = torch.tensor([[100, 1]], dtype=torch.float32)
ty = torch.tensor([[600]], dtype=torch.float32)

predicted_price = model(tx)

print(f"\nPredicted price: {predicted_price.item():.2f}")
print(f"Actual target price: {ty.item():.2f}")
print(f"Difference (error): {abs(predicted_price.item() - ty.item()):.2f}")
