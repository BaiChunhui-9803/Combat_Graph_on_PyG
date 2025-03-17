from MyGNNs import MyGCN, MyGAT
import wandb
from torch_geometric.transforms import RandomNodeSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv
import wandb
import json
import os
from MyGNNs import MyGCN, MyGAT, MyGraphSAGE
import pandas as pd
import numpy as np


def embedding_to_wandb(h, color, key="embedding"):
    num_components = h.shape[-1]
    df = pd.DataFrame(data=h.detach().cpu().numpy(),
                      columns=[f"c_{i}" for i in range(num_components)])
    df["target"] = color.detach().cpu().numpy().astype("str")
    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]
    wandb.log({key: df})

def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean().item()

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_acc = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)  # 传递 batch 参数
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_acc += accuracy(out, data.y) * data.num_graphs
    train_loss = total_loss / len(train_loader.dataset)
    train_acc = total_acc / len(train_loader.dataset)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    return total_loss / len(train_loader.dataset), total_acc / len(train_loader.dataset)


def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data in test_loader:
            out = model(data.x, data.edge_index, data.batch)  # 传递 batch 参数
            loss = criterion(out, data.y)

            total_loss += loss.item() * data.num_graphs
            total_acc += accuracy(out, data.y) * data.num_graphs
    test_loss = total_loss / len(test_loader.dataset)
    test_acc = total_acc / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    return total_loss / len(test_loader.dataset), total_acc / len(test_loader.dataset)

def execute_sweep(state_graphs, num_label):
    pro_name = "micro_graph_classification"
    model_name = "GraphSAGE"
    train_graphs, test_graphs = state_graphs[:int(0.9 * len(state_graphs))], state_graphs[int(0.9 * len(state_graphs)):]
    train_loader = DataLoader(train_graphs, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1024, shuffle=False)

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "lr": {"distribution": "uniform", "min": 0.001, "max": 0.1},
            "hidden_channels": {"values": [64, 128, 256]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="graph_sage_sweep")

    def train_with_wandb():
        run = wandb.init()
        config = wandb.config
        model = MyGraphSAGE(in_channels=state_graphs[0].num_node_features,
                            hidden_channels=config.hidden_channels,
                            out_channels=num_label)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        # print(f"Log embedding_before_training")
        #
        # with torch.no_grad():
        #     for data in test_loader:
        #         h = model(data.x, data.edge_index, data.batch)  # 获取嵌入
        #         embedding_to_wandb(h, data.y, key="embedding_before_training")

        print(f"Starting run with lr={config.lr} and hidden_channels={config.hidden_channels}")

        for epoch in range(50):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion)
            test_loss, test_acc = test(model, test_loader, criterion)

            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            })

        # print(f"Log embedding_after_training")
        # with torch.no_grad():
        #     for data in test_loader:
        #         h = model(data.x, data.edge_index, data.batch)  # 获取嵌入
        #         embedding_to_wandb(h, data.y, key="embedding_after_training")

        sum_test_loss, sum_test_acc = test(model, test_loader, criterion)
        wandb.summary[f"accuracy"] = sum_test_acc

        print(f"Run finished with Test Accuracy: {sum_test_acc:.4f}")
        run.finish()

    wandb.agent(sweep_id, train_with_wandb, count=200)

    # transform = RandomNodeSplit(num_val=0.0, num_test=0.1)
    # data = transform(state_graphs[0])

    model_map = {
        # "GCN": "MyGCN",
        # "MLP": "MyMLP",
        # "GAT": "MyGAT",
        # "Transformer": "MyTransformer",
        # "GatedGCN": "MyGatedGCN",
        # "GIN": "GINConv",
        "GraphSAGE": "MyGraphSAGE",
        # "GatedGCN": "GatedGraphConv",
        # "APPNP": "APPNP",
        # "GraphConv": "GraphConv"
    }






    pass