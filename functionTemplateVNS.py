import torch
import torch.nn as nn
from torch.optim import Optimizer
import copy

class VNSOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, neighborhood_size=0.1, max_iter=100):
        defaults = dict(lr=lr, neighborhood_size=neighborhood_size, max_iter=max_iter)
        super(VNSOptimizer, self).__init__(params, defaults)

    def step(self, model, criterion, val_loader, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Save original parameters
                original_params = copy.deepcopy(p.data)

                # Perturb the parameters within the neighborhood
                neighborhood = torch.randn_like(p.data) * group['neighborhood_size']
                p.data.add_(neighborhood)

                # Evaluate the perturbed model
                perturbed_loss = self.evaluate_model(model, criterion, val_loader)

                # If the new parameters are worse, revert to the original
                if perturbed_loss > loss:
                    p.data = original_params
                
                # Else, keep the perturbed parameters
                else:
                    loss = perturbed_loss

        return loss

    def evaluate_model(self, model, criterion, val_loader):
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        model.train()
        return val_loss / len(val_loader)

# Assuming you have a U-Net model defined as a class
model = UNet()

# Custom loss function suitable for segmentation
criterion = nn.CrossEntropyLoss()

# Example data loader for validation
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# VNS optimizer
optimizer = VNSOptimizer(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step(model, criterion, val_loader)
