from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader,Subset
from torchvision.models import alexnet, AlexNet_Weights


from torchmetrics.classification import  MulticlassAccuracy,MulticlassPrecision, MulticlassRecall, MulticlassF1Score
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


import json
from pathlib import Path
def save_model_results(model, results, save_dir, model_name):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        save_dir / f"{model_name}_weights.pth",
    )

    scores = {
        name: score.item() * 100
        for name, score in results.items()
    }

    with open(save_dir / f"{model_name}_results.json", "w") as file:
        json.dump(scores, file, indent=4)


def train_model(model, train_loader,loss_function,optimizer,device):
    model.train() # SET MODE TO TRAIN
    running_loss = 0.0
    for images,labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        #Tracking progress
        running_loss += loss.item()

    # AFTER all batches are finished
    avg_loss = running_loss / len(train_loader)
    print(f"Loss: {avg_loss:.2f}")


def evaluate_metrics(model, dataloader, device, num_classes=10):
    model.eval()

    metrics = {
        'accuracy': MulticlassAccuracy(num_classes=num_classes, average='micro').to(device),
        'precision': MulticlassPrecision(num_classes=num_classes, average='macro').to(device),
        'recall': MulticlassRecall(num_classes=num_classes, average='macro').to(device),
        'f1': MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    }

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            for metric in metrics.values():
                metric.update(outputs, labels)
    return {
        name: metric.compute()
        for name, metric in metrics.items()
    }


def model_epochs(model,train_loader,loss_function,optimizer,device,eval_loader, num_epochs =5,scheduler=None, start_epoch=0):
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch + 1}")

        train_model(
            model, train_loader, loss_function, optimizer, device
        )

        results = evaluate_metrics(model, eval_loader, device)
        scheduler.step(results["accuracy"].item())
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.8f}")

        for name, score in results.items():
            print(f"{name}: {score.item() * 100:.2f}%")




def inference_from_alexnext(image,device):
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()  # automate transformation of data

    # image.show()
    # Prepare the image and add the batch dimension
    # shape (1,3,224,224) the 1 is batch size
    inputs = preprocess(image).unsqueeze(0).to(device)

    # Predict without calculating gradients
    with torch.no_grad():
        output = model(inputs)
        probabilities = output.softmax(dim=1)

    confidence, predicted_class = probabilities[0].max(dim=0)  # find the highest among the 1000 classes
    class_name = weights.meta["categories"][predicted_class.item()]

    print(f"Prediction: {class_name}")
    print(f"Model probability: {confidence.item():.2%}")
    return model


def dataset_from_transform(train_transform, eval_transform):
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
    eval_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=eval_transform)

    return train_dataset, eval_dataset,test_dataset



def dataloader_from_dataset(train_dataset, eval_dataset,test_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader,test_loader


def dataset_split(indices, train_dataset, eval_dataset):
    train_dataset_full = Subset(train_dataset, indices[:45000])
    eval_dataset_full = Subset(eval_dataset, indices[45000:])
    return train_dataset_full, eval_dataset_full