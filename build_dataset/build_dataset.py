import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import wandb
import json

# =============================================================================

pro_name = "sce2m2train2"

# =============================================================================
# 配置W&B
use_wandb = True #@param {type:"boolean"}
use_wandb = False
wandb_project = "build_dataset" #@param {type:"string"}
wandb_run_name = f"{pro_name}_dataset_builder" #@param {type:"string"}
data_set_path = f"D:/OFEC/ofec/instance/problem/realworld/game/data_set"

if use_wandb:
    wandb.init(project=wandb_project, name=wandb_run_name)
# =============================================================================
# =============================================================================
# 加载数据集
from load_dataset import mydataset
# data_set_path = "D:/OFEC/ofec/instance/problem/realworld/game/data_set"

my_data_set = mydataset(root=data_set_path, name=pro_name)
data_set = my_data_set.get(0)
print(data_set)

data_details = {
    "x": data_set.x.t().tolist(),
    "edge_index": data_set.edge_index.tolist(),
    "y": data_set.y.tolist(),
    "num_nodes": data_set.num_nodes,
    "num_node_features": data_set.num_node_features,
    "num_edges": data_set.num_edges,
    "num_edge_features": data_set.num_edge_features,
    "num_classes": my_data_set.num_classes,
    "avg_node_degree": data_set.num_edges / data_set.num_nodes,
    "has_isolated_nodes": data_set.has_isolated_nodes(),
    "has_self_loops": data_set.has_self_loops(),
    "is_undirected": data_set.is_undirected()
}

if use_wandb:
    wandb.log({"data_details": data_details})
    dataset_artifact = wandb.Artifact(name=pro_name, type="dataset", metadata=data_details)
    dataset_artifact.add_dir(data_set_path)
    wandb.log_artifact(dataset_artifact)
else:
    print(json.dumps(data_details, sort_keys=True, indent=4))

# =============================================================================
data = RandomNodeSplit(num_test=0.3)(data_set)

wandb.finish()