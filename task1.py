import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.geo_triplet_dataset import GeoTripletDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
model = model.to(device)

# Option : only train the head by freezing the backbone
if hasattr(model, "backbone"):
    for p in model.backbone.parameters():
        p.requires_grad = False


transform = transforms.Compose([
    transforms.Resize((322, 322)),
    transforms.ToTensor(),
])


train_ds = GeoTripletDataset(
    annotations_path="dataset/splits/annotations_train.json",
    image_dir="/scratch/users/agraillet/images",
    transform=transform,
)

val_ds = GeoTripletDataset(
    annotations_path="dataset/splits/annotations_val.json",
    image_dir="/scratch/users/agraillet/images",
    transform=transform,
)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)


criterion = nn.TripletMarginLoss(margin=0.2)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)

num_epochs = 5

for epoch in range(num_epochs):
    # --------- train ---------
    model.train()
    total_train_loss = 0.0
    n_train_batches = 0

    for anc, pos, neg in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", unit="batch"
    ):
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

        f_anc = model(anc)
        f_pos = model(pos)
        f_neg = model(neg)

        loss = criterion(f_anc, f_pos, f_neg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        n_train_batches += 1

    avg_train_loss = total_train_loss / max(1, n_train_batches)

    # --------- validation ---------
    model.eval()
    total_val_loss = 0.0
    n_val_batches = 0

    with torch.no_grad():
        for anc, pos, neg in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]", unit="batch"
        ):
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

            f_anc = model(anc)
            f_pos = model(pos)
            f_neg = model(neg)

            loss = criterion(f_anc, f_pos, f_neg)
            total_val_loss += loss.item()
            n_val_batches += 1

    avg_val_loss = total_val_loss / max(1, n_val_batches)

    print(
        f"Epoch {epoch+1}/{num_epochs} - "
        f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}"
    )