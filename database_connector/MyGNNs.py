import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GATv2Conv, GCN2Conv, global_mean_pool

class MyGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MyGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 图卷积层
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # 全局池化层
        x = global_mean_pool(x, batch)

        # 全连接层
        x = self.fc(x)
        return x

class MyGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MyGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)  # 使用多头注意力
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=1, concat=False, dropout=0.6)  # 合并多头注意力
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)  # 全局池化
        x = self.fc(x)  # 全连接层
        return x

class MyGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MyGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)  # 全局池化
        x = self.fc(x)  # 全连接层
        return x

class MyGATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MyGATv2, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * 8, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)  # 全局池化
        x = self.fc(x)  # 全连接层
        return x

class MyGCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(MyGCNII, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha=0.1, theta=0.5, layer=i + 1))
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.lin1(x))
        x0 = x
        for conv in self.convs:
            x = F.relu(conv(x, x0, edge_index))
        x = global_mean_pool(x, batch)  # 全局池化
        x = self.lin2(x)  # 全连接层
        return x