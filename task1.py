import torch
model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
model.eval()
