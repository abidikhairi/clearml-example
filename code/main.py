import torch as th
from torch.utils.data import DataLoader
from torchvision import transforms as T 
from torchvision import datasets
from torchmetrics import Accuracy

from models.linear import LinearMNISTModel
from utils.cli_parser import get_default_parser
from utils.logging import get_default_logger


def main(args):
    logger = get_default_logger(__name__, 'logs/main.log')
    
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambd=lambda x: x.view(-1))
    ])

    train_mnist = datasets.MNIST(root=args.data_root, train=True, transform=transform, download=True)
    test_mnist = datasets.MNIST(root=args.data_root, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_mnist, batch_size=args.batch_size)
    train_loader = DataLoader(test_mnist, batch_size=args.batch_size)
    
    model = LinearMNISTModel()
    optimizer = th.optim.Adam(model.parameters(), lr=0.0003)
    criterion = th.nn.NLLLoss()

    accuracy = Accuracy(task="multiclass", num_classes=10)

    model.train()

    logger.info(f"Start training for {args.num_epochs}")
    
    for epoch in range(args.num_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % args.log_interval == 0:
                logger.info(f'Epoch [{epoch + 1}/{args.num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        logger.info(f'Epoch [{epoch + 1}/{args.num_epochs}] Starting evaluation')
        with th.no_grad():
            accs = []
            for batch_idx, (images, labels) in enumerate(train_loader):
                outputs = model(images)

                batch_accuracy = accuracy(th.exp(outputs), labels).item()
                accs.append(batch_accuracy)
            
            val_accuracy = th.tensor(accs).mean().item() * 100

            logger.info(f'Epoch [{epoch + 1}/{args.num_epochs}] Accuracy: {val_accuracy:.4f} %')


if __name__ == '__main__':
    parser = get_default_parser()

    args = parser.parse_args()
    
    main(args)
