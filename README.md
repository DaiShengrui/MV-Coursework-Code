# MV-Coursework-Code
Dual CNN with Attention

Custom precision function

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

Custom loss function

(Used in model)

Cross Entropy Loss
ce_loss = nn.CrossEntropyLoss()
# outputs size (batch_size, num_classes)，targets size (batch_size)
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.tensor([1, 0, 4], dtype=torch.int64)
loss = ce_loss(outputs, targets)


(Unused in model)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBinaryLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, reduction='mean'):
        """
        Custom binary loss function
        :param alpha: Coefficient for balancing cross-entropy loss and L2 regularization
        :param beta: Coefficient for balancing cross-entropy loss and L1 regularization
        :param reduction: Specifies the method to reduce the loss, either 'mean' or 'sum'
        """
        super(CustomBinaryLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, outputs, targets):
        """
        Compute the loss function
        :param outputs: Output of the model
        :param targets: Real labels
        :return: Computed loss
        """
        # Cross-entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction=self.reduction)

        # L2 regularization term
        l2_reg = torch.tensor(0.).to(outputs.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)

        # L1 regularization term
        l1_reg = torch.tensor(0.).to(outputs.device)
        for param in self.parameters():
            l1_reg += torch.norm(param, p=1)

        # Combined loss
        loss = ce_loss + self.alpha * l2_reg + self.beta * l1_reg
        return loss







