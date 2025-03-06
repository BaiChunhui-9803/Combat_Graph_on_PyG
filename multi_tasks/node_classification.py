import pandas
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import json
import yaml
from build_dataset.load_dataset import mydataset
import os
import wandb
from torch_geometric.transforms import RandomNodeSplit
# =============================================================================

# enable_wandb = False
enable_wandb = True
# enable_sweep = False
enable_sweep = True
DEBUG = True
pro_name = "sce2m2train2"
model_name = "GAT"

# =============================================================================
with open(f'config/{model_name.lower()}_run_config.yaml', 'r') as file:
    run_config = yaml.safe_load(file)

run_name_model = run_config['model']
run_name_test_size = f"T{int(run_config['test_size']*10)}"
run_name_layers = f"L{run_config['layers']}"
run_name_hidden_channels = f"H{run_config['hidden_channels']}"
run_name_lr = f"LR{run_config['lr']}"
run_name_weight_decay = f"WD{run_config['weight_decay']}"
run_name_momentum = f"M{run_config['momentum']}"
run_name = f"{run_name_model}{run_name_test_size}{run_name_layers}{run_name_hidden_channels}{run_name_lr}{run_name_weight_decay}{run_name_momentum}"

model_map = {
    "GCN": "MyGCN",
    "MLP": "MyMLP",
    "GAT": "MyGAT",
    "Transformer": "MyTransformer",
    # "GIN": "GINConv",
    # "GraphSAGE": "SAGEConv",
    # "GatedGCN": "GatedGraphConv",
    # "APPNP": "APPNP",
    # "GraphConv": "GraphConv"
}

# 全局函数
def embedding_to_wandb(h, color, key="embedding"):
    num_components = h.shape[-1]
    df = pandas.DataFrame(data=h.detach().cpu().numpy(),
                        columns=[f"c_{i}" for i in range(num_components)])
    df["target"] = color.detach().cpu().numpy().astype("str")
    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]
    wandb.log({key: df})

def visualize_to_wandb(h, color, method):
    h_np = h.detach().cpu().numpy()
    n_samples = h_np.shape[0]
    z = TSNE(n_components=2, perplexity=min(30, n_samples // 3)).fit_transform(h_np)
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    title = method + "/embedding_tSNE/trained"
    wandb.log({
        title: wandb.Plotly(plt.gcf())
    })

# =============================================================================
# 配置W&B/加载数据集
# wandb.login(key='7f6d598bc0f0aca4dc78e698dabfcdb76e832a19')

folder_path = f"artifacts/{pro_name}-v0"
dataset_dir = None
if enable_wandb:
    run = wandb.init(project=pro_name, name=run_name)

if os.path.exists(folder_path) and os.path.isdir(folder_path) and DEBUG:
    dataset_dir = folder_path
else:
    if enable_wandb:
        run = wandb.init(project=pro_name, name=run_name)
        artifact = run.use_artifact(f'bch980311-/build_dataset/{pro_name}:v0', type='dataset')
        dataset_dir = artifact.download()
# =============================================================================
# =============================================================================
# 数据集随机分割
# 获取掩码张量
#   train - data.train_mask
#   val   - data.val_mask
#   test  - data.test_mask
torch.manual_seed(run_config['seed'])

my_data_set = mydataset(root=dataset_dir, name=pro_name)
data_set = my_data_set.get(0)
# if data_set.y.dtype != torch.int32:
data_set.x = torch.tensor(data_set.x, dtype=torch.float)
data_set.y = torch.tensor(data_set.y, dtype=torch.int64)
data = RandomNodeSplit(split='train_rest',
                       num_val=0.0,
                       num_test=run_config["test_size"])(data_set)

data_details = {
    "num_train_nodes": data.train_mask.sum(),
    "rate_train_nodes": int(data.train_mask.sum()) / data.num_nodes,
    "train_mask": data.train_mask.tolist(),
    "val_mask": data.val_mask.tolist(),
    "test_mask": data.test_mask.tolist()
}

if enable_wandb:
    wandb.summary["data_details"] = data_details
# =============================================================================
# =============================================================================
# 构建GCN
import importlib
import GNNs.MyGNNs as my_gnns

ModelClass = getattr(my_gnns, model_map[run_name_model])

model = ModelClass(layers = run_config["layers"],
              in_channels = data.num_node_features,
              hidden_channels = run_config["hidden_channels"],
              out_channels = my_data_set.num_classes)
print(model)

model.eval()

out = model(x=data.x, edge_index=data.edge_index)

if enable_wandb:
    embedding_to_wandb(out, color=data.y, key="gcn/embedding/init")

# =============================================================================
# =============================================================================
# 训练GCN
if enable_wandb:
    wandb.watch(model)

optimizer = torch.optim.SGD(model.parameters(),
                            lr=run_config["lr"],
                            weight_decay=run_config["weight_decay"],
                            momentum=run_config["momentum"])
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(x=data.x, edge_index=data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test():
    model.eval()
    out = model(x=data.x, edge_index=data.edge_index)
    pred = out.argmax(dim=-1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


for epoch in range(1, 500):
    loss = train()
    test_acc = test()
    if enable_wandb:
        wandb.log({"gcn/loss": loss, "gcn/accuracy": test_acc})
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {test_acc:.4f}')

model.eval()

out = model(x=data.x, edge_index=data.edge_index)

test_acc = test()
if enable_wandb:
    wandb.summary["gcn/accuracy"] = test_acc
    embedding_to_wandb(out, color=data.y, key="gcn/embedding/trained")
    visualize_to_wandb(out, data.y, 'gcn')
    wandb.finish()

if enable_wandb:
    wandb.finish()
# =============================================================================
# 超参数搜索
import tqdm
def agent_fn():
    wandb.init()
    torch.manual_seed(wandb.config.seed)
    # data = RandomNodeSplit(split='train_rest',
    #                        num_val=0.0,
    #                        num_test=0.1)(data_set)
    # data = RandomNodeSplit(split='train_rest',
    #                        num_val=0.0,
    #                        num_test=wandb.config.test_size)(data_set)
    model = ModelClass(layers=wandb.config.layers,
                  in_channels = data.num_node_features,
                  hidden_channels = wandb.config.hidden_channels,
                  out_channels = my_data_set.num_classes)
    wandb.watch(model)

    with torch.no_grad():
      out = model(x=data.x, edge_index=data.edge_index)
      embedding_to_wandb(out, color=data.y, key=f"{model_name.lower()}/embedding/init")

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=wandb.config.lr,
                                weight_decay=wandb.config.weight_decay,
                                momentum=wandb.config.momentum)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
          model.train()
          optimizer.zero_grad()  # Clear gradients.
          out = model(x=data.x, edge_index=data.edge_index)  # Perform a single forward pass.
          loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
          loss.backward()  # Derive gradients.
          optimizer.step()  # Update parameters based on gradients.
          return loss

    def test():
          model.eval()
          with torch.no_grad():
              out = model(x=data.x, edge_index=data.edge_index)
              pred = out.argmax(dim=-1)  # Use the class with highest probability.
              test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
              test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
              test_loss = criterion(out[data.test_mask], data.y[data.test_mask])  # Compute test loss
          return test_acc

    for epoch in tqdm.tqdm(range(1, 500)):
        loss = train()
        test_acc = test()
        wandb.log({f"{model_name.lower()}/loss": loss, f"{model_name.lower()}/accuracy": test_acc})

    model.eval()

    out = model(x=data.x, edge_index=data.edge_index)
    test_acc = test()
    wandb.summary[f"{model_name.lower()}/accuracy"] = test_acc
    # wandb.log({"gcn/accuracy": test_acc})
    embedding_to_wandb(out, color=data.y, key=f"{model_name.lower()}/embedding/trained")
    wandb.finish()

with open(f'config/{model_name.lower()}_sweep_config.yaml', 'r') as file:
    sweep_config = yaml.safe_load(file)

if enable_sweep:
    sweep_id = wandb.sweep(sweep_config, project=f"{pro_name}_sweep")
    wandb.agent(sweep_id, project=f"{pro_name}_sweep", function=agent_fn, count=1000)



wandb.finish()