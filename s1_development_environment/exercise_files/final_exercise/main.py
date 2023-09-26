import click
import torch
from model import MyAwesomeModel
from torch import nn

from data import mnist


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    epochs = 5
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)


    train_set, _ = mnist()

    for ep in range(1,epochs+1):
        total_loss = 0
        num_correct = 0
        for batch_idx, (x,y) in enumerate(train_set):
            x = x.to("cpu")
            y = y.to("cpu")

            model.train()
            y_hat = model(x)
            batch_loss = loss(y_hat,y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('BATCH:\t({:5} / {:5})\tLOSS:\t{:.3f}'
                .format(batch_idx, float(batch_loss)), end='\r')

        total_loss += float(batch_loss)
        num_correct += int(torch.sum(torch.argmax(y_hat, dim=1) == y))

    print('EPOCH:\t{:5}\tLOSS:\t{:.3f}\tACCURACY:\t{:.3f}'
        .format(ep, total_loss / len(train_set), num_correct / len(train_set ),
                end='\r'))


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    pass
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
