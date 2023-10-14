import os
import clearml
import torch as th
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T 
from torchvision import datasets
from torchmetrics import Accuracy, ConfusionMatrix
from clearml import Task, Logger
from tempfile import gettempdir

from models.linear import LinearMNISTModel
from utils.cli_parser import get_default_parser
from utils.logging import get_default_logger


def main(args):
    task = Task.init(project_name='toy_examples', task_name='mnist')

    logger = get_default_logger(__name__)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambd=lambda x: x.view(-1))
    ])

    train_mnist = datasets.MNIST(root=args.data_root, train=True, transform=transform, download=True)
    test_mnist = datasets.MNIST(root=args.data_root, train=False, transform=transform, download=True)
    
    num_test_examples = len(test_mnist)
    num_validation_examples = int(num_test_examples - (num_test_examples * 0.5))
    valid_mnist, test_mnist = random_split(test_mnist, [num_validation_examples, num_test_examples - num_validation_examples])

    train_loader = DataLoader(train_mnist, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_mnist, batch_size=args.batch_size)
    test_loader = DataLoader(test_mnist, batch_size=args.batch_size)
    
    model = LinearMNISTModel()
    optimizer = th.optim.Adam(model.parameters(), lr=0.0003)
    criterion = th.nn.NLLLoss()

    accuracy = Accuracy(task="multiclass", num_classes=10)
    confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=10)

    model.train()

    logger.info(f"Start training for {args.num_epochs}")
    prev_acc = -1
    
    for epoch in range(args.num_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % args.log_interval == 0:
                logger.info(f'Epoch [{epoch + 1}/{args.num_epochs}] Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                Logger.current_logger().report_scalar(title="Training Loss", series="train_loss", value=loss.item(), iteration=batch_idx)
                

        logger.info(f'Epoch [{epoch + 1}/{args.num_epochs}] Starting evaluation')
        with th.no_grad():
            accs = []
            for batch_idx, (images, labels) in enumerate(valid_loader):
                outputs = model(images)

                batch_accuracy = accuracy(th.exp(outputs), labels).item()
                accs.append(batch_accuracy)
            
            val_accuracy = th.tensor(accs).mean().item() * 100

            Logger.current_logger().report_scalar(title="Validation Accuracy", series='val_accuracy', value=val_accuracy, iteration=epoch)
            logger.info(f'Epoch [{epoch + 1}/{args.num_epochs}] Accuracy: {val_accuracy:.4f} %')

            if val_accuracy > prev_acc:
                logger.info("Saving new Model to local tempdir")
                prev_acc = val_accuracy

                th.save(model.state_dict(), os.path.join(gettempdir(), "mnist_mlp.pt"))

    logger.info("Training finished")
    model.eval()
    
    with th.no_grad():
        targets = []
        preds = []
        for batch_idx, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            pred = th.argmax(th.exp(outputs), dim=-1).flatten().tolist()

            targets.extend(labels.flatten().tolist())
            preds.extend(pred)
        
        cm = confusion_matrix(th.tensor(preds), th.tensor(targets)).numpy()
        Logger.current_logger().report_confusion_matrix(title="Confusion Matrix", series="Classes", matrix=cm)


if __name__ == '__main__':
    parser = get_default_parser()

    args = parser.parse_args()
    
    main(args)
