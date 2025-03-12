import torch
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time
import argparse
from modified_squeezenet import ModifiedSqueezeNet  # Adjust this import based on your file structure

# Argument parsing
parser = argparse.ArgumentParser(description='Test SqueezeNet on CIFAR-10')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--model_name', type=str, required=True, help='Path to the saved model weights')
parser.add_argument('--num_classes', type=int, default=131)  # Number of classes
parser.add_argument('--no-cuda', action='store_true', default=False)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# DataLoader setup
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224),
                             (0.247032237587, 0.243485133253, 0.261587846975))
    ])),
    batch_size=args.batch_size, shuffle=False, **kwargs)

# Initialize model
net = ModifiedSqueezeNet(num_classes=args.num_classes)
if args.model_name is not None:
    print("Loading pre-trained weights")
    pretrained_weights = torch.load(args.model_name)
    net.load_state_dict(pretrained_weights)

if args.cuda:
    net.cuda()

def top_k_accuracy(scores, targets, k=5):
    """Calculate top-k accuracy."""
    _, top_k_preds = scores.topk(k, dim=1, largest=True, sorted=True)
    correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
    return correct.any(dim=1).float().sum().item() / len(targets)

def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_targets = []
    all_predictions = []
    all_scores = []
    start_time = time.time()

    with torch.no_grad():  # No need to track gradients during testing
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            scores = model(data)
            _, predictions = torch.max(scores, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.append(scores.cpu())



    # Concatenate all scores for top-5 accuracy
    all_scores = torch.cat(all_scores, dim=0)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    top1_accuracy = accuracy
    top5_accuracy = top_k_accuracy(all_scores, torch.tensor(all_targets), k=5)


    end_time = time.time()
    time_elapsed = end_time - start_time

    print(f'Accuracy: {accuracy:.6f}')
    print(f'Precision: {precision:.6f}')
    print(f'Recall: {recall:.6f}')
    print(f'F1-score: {f1:.6f}')
    print(f'Top-1 Accuracy: {top1_accuracy:.6f}')
    print(f'Top-5 Accuracy: {top5_accuracy:.6f}')
    print(f'Time: {time_elapsed:.2f}s')

    # Calculate model parameters (M)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total model parameters (M): {total_params / 1e6:.2f}')  # Convert to millions

    return accuracy, precision, recall, f1, top1_accuracy, top5_accuracy, time_elapsed, total_params / 1e6

if __name__ == '__main__':
    # Run the test
    test_metrics = test(net, test_loader, device='cuda' if args.cuda else 'cpu')
    print(f"Testing results: {test_metrics}")
