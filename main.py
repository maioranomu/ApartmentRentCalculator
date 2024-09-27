import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
device = torch.device("cuda:0")
if torch.cuda.is_available():
     torch.set_default_device(device)

directory = "Memory3"
model_path = os.path.join(f"{directory}/model.pth")
optimizer_path = os.path.join(f"{directory}/optimizer.pth")
if not os.path.exists(directory):
    os.makedirs(directory)
if not os.path.exists(model_path):
    open(model_path, 'w').close()
if not os.path.exists(optimizer_path):
    open(optimizer_path, 'w').close()
"""
Region:
1 : 6x
2 : 3x
3 : 1x
"""

examples = 250
df = []
for i in range(examples):
    price = 0
    m2 = random.randint(1, 501)
    region = random.randint(1, 4)
    if region == 1:
        price = m2 * 6
    elif region == 2:
        price = m2 * 3
    else:
        price = m2
    df.append([m2, region, price])
# print(df)
print("\n")

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

        if epoch == 40000:
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

        #print(f"Epoch [{epoch + 1}/{epochs}] | Loss: [{loss.item():.4f}]")
        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: [{loss.item():.4f}]")
            if loss.item() <= 0.0001:
                break
    predictions = model(x)
    final_loss = loss_fn(predictions, y)
    print(f"\nFinal Loss after training: {final_loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    print("Saved")

#------------------------------------------
tm2 = 177
tr = 1
t_price = 0
if tr == 1:
    t_price = tm2 * 6
elif region == 2:
    tr = tm2 * 3
else:
    t_price = m2
tx = torch.tensor([[tm2, tr]], dtype=torch.float32)
ty = torch.tensor([[t_price]], dtype=torch.float32)

predicted_price = model(tx)

print(f"\nPredicted price: {predicted_price.item():.2f}")
print(f"Actual target price: {ty.item():.2f}")
print(f"Difference (error): {abs(predicted_price.item() - ty.item()):.2f}")
