import torch
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, SAGEConv, TransformerConv, ChebConv
from torch_geometric.nn import GraphConv, GINConv, EdgeConv, DynamicEdgeConv, GATv2Conv

# 多层感知机
class MyMLP(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.layers = layers
        self.lin1 = Linear(in_channels, hidden_channels)
        for i in range(2, layers):
            setattr(self, f'lin{i}', Linear(hidden_channels, hidden_channels))
        setattr(self, f'lin{layers}', Linear(hidden_channels, out_channels))

    def forward(self, x, **kwargs):
        for i in range(self.layers):
            x = getattr(self, f'lin{i + 1}')(x)
            if i != self.layers - 1:
                x = x.relu()
                if i == 3:
                    x = F.dropout(x, p=0.5, training=self.training)
        return x

# 图卷积
class MyGCN(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        return x

# 图注意力
class MyGAT(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super(MyGAT, self).__init__()
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.convs = ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=0.0))
        for i in range(layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.0))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, dropout=0.0))

    def forward(self, x, edge_index, **kwargs):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x

# Transformer
class MyTransformer(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super(MyTransformer, self).__init__()
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.convs = ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=0.0))
        for i in range(layers - 2):
            self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.0))
        self.convs.append(TransformerConv(hidden_channels * heads, out_channels, heads=1, dropout=0.0))

    def forward(self, x, edge_index, **kwargs):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x

class MyGatedGCN(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels):
        super(MyGatedGCN, self).__init__()
        self.layers = layers
        self.gated_conv = GatedGraphConv(out_channels=out_channels, num_layers=layers)
        self.fc = Linear(out_channels, out_channels)

    def forward(self, x, edge_index, **kwargs):
        x = self.gated_conv(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

class MySAGE(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels):
        super(MySAGE, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

class MyCheb(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels, K=2):
        super(MyCheb, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(ChebConv(in_channels, hidden_channels, K))
        for _ in range(layers - 2):
            self.convs.append(ChebConv(hidden_channels, hidden_channels, K))
        self.convs.append(ChebConv(hidden_channels, out_channels, K))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

class MyGraphConv(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels):
        super(MyGraphConv, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels))
        for _ in range(layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
        self.convs.append(GraphConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

class MyGIN(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels):
        super(MyGIN, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(Linear(in_channels, hidden_channels)))
        for _ in range(layers - 2):
            self.convs.append(GINConv(Linear(hidden_channels, hidden_channels)))
        self.convs.append(GINConv(Linear(hidden_channels, out_channels)))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

class MyEdgeConv(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels):
        super(MyEdgeConv, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(EdgeConv(Linear(2 * in_channels, hidden_channels)))
        for _ in range(layers - 2):
            self.convs.append(EdgeConv(Linear(2 * hidden_channels, hidden_channels)))
        self.convs.append(EdgeConv(Linear(2 * hidden_channels, out_channels)))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

class MyDynamicEdgeConv(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels, k=2):
        super(MyDynamicEdgeConv, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(DynamicEdgeConv(Linear(2 * in_channels, hidden_channels), k=k))
        for _ in range(layers - 2):
            self.convs.append(DynamicEdgeConv(Linear(2 * hidden_channels, hidden_channels), k=k))
        self.convs.append(DynamicEdgeConv(Linear(2 * hidden_channels, out_channels), k=k))

    def forward(self, x, batch):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, batch)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

class MyGATv2(torch.nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels, heads=4):
        super(MyGATv2, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads))
        for _ in range(layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads))
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x