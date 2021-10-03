import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """
    An umbrella model for both linear, logistic and softmax regression
    linear regression can be trained using MSELoss, L1Loss etc
    logistic regression can be trained using BCELoss.
    softmax regression can be trained using CrossEntropyLoss, nllLoss
    """

    def __init__(self, in_units: int, out_units: int):
        """
        Initializes the parameters

        :param in_units: feature dimensions
        :param out_units: output units (n_classes)
        """
        super().__init__()
        self.linear_model = nn.Linear(in_units, out_units)

    def forward(self, x: torch.Tensor):
        return self.linear_model(x).squeeze()

    @torch.no_grad()
    def predict(self, data: torch.Tensor):
        """
        Returns the predictions on the data
        :param data: a torch tensor of shape (m, n) m => n_examples; n => feature dims
        :return: predictions on the data of shape (m, out_units)
        """
        self.eval()  # Put the network in eval mode
        predictions = self(data)
        return predictions

    @torch.no_grad()
    def calculate_loss(self, data: torch.Tensor, targets: torch.Tensor, loss_fn: nn.Module):
        """
        Calculates and detaches the loss
        :param data: the data to make the predictions on
        :param targets: the ground truth labels
        :param loss_fn: the loss function to be used
        :return: returns the detached loss wrt the labels
        """
        predictions = self.predict(data)
        loss = loss_fn(predictions, targets)
        return loss.detach()
