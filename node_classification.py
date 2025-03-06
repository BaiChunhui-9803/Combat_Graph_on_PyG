import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


# =============================================================================
# 加载数据集
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# =============================================================================
# 构建一个MLP
import torch
from torch.nn import Linear
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

model = MLP(hidden_channels=16)
print(model)

# =============================================================================
# 训练一个MLP
from IPython.core.display_functions import display
from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = MLP(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

# 初始化记录损失和准确率的列表
train_losses = []
test_accuracies = []
for epoch in range(1, 201):
    loss = train()  # 假设train()函数返回当前epoch的损失
    train_losses.append(loss.item())  # 记录损失
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc = test()  # 假设test()函数返回当前epoch的测试准确率
    test_accuracies.append(test_acc)  # 记录测试准确率
    print(f'Test Accuracy: {test_acc:.4f}')

# 创建图和轴对象
fig, ax1 = plt.subplots(figsize=(10, 5))
# 绘制训练损失曲线
ax1.plot(train_losses, label='Training Loss', color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
# 创建第二个y轴
ax2 = ax1.twinx()
# 绘制测试准确率曲线
ax2.plot(test_accuracies, label='Test Accuracy', color='red')
ax2.set_ylabel('Test Accuracy', color='red')
ax2.tick_params(axis='y', labelcolor='red')
# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# 添加标题
plt.title('Training Loss and Test Accuracy')
# 显示图表
plt.show()

# =============================================================================
# 构建一个GCN
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
print(model)

# =============================================================================
# 可视化未经训练的GCN网络的节点嵌入
model = GCN(hidden_channels=16)
model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)

# =============================================================================
# 训练一个GCN
model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

# 初始化记录损失和准确率的列表
train_losses = []
test_accuracies = []
for epoch in range(1, 201):
    loss = train()  # 假设train()函数返回当前epoch的损失
    train_losses.append(loss.item())  # 记录损失
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc = test()  # 假设test()函数返回当前epoch的测试准确率
    test_accuracies.append(test_acc)  # 记录测试准确率
    print(f'Test Accuracy: {test_acc:.4f}')

# 创建图和轴对象
fig, ax1 = plt.subplots(figsize=(10, 5))
# 绘制训练损失曲线
ax1.plot(train_losses, label='Training Loss', color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
# 创建第二个y轴
ax2 = ax1.twinx()
# 绘制测试准确率曲线
ax2.plot(test_accuracies, label='Test Accuracy', color='red')
ax2.set_ylabel('Test Accuracy', color='red')
ax2.tick_params(axis='y', labelcolor='red')
# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# 添加标题
plt.title('Training Loss and Test Accuracy')
# 显示图表
plt.show()

# =============================================================================
# 可视化训练模型的输出嵌入
model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)

# =============================================================================
# 改进GNN，将GCNConv替换为GATConv
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(heads * hidden_channels, dataset.num_classes, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return x

model = GAT(hidden_channels=8, heads=8)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def test(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    return acc

# for epoch in range(1, 201):
#     loss = train()
#     val_acc = test(data.val_mask)
#     test_acc = test(data.test_mask)
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

# 初始化记录损失和准确率的列表
train_losses = []
test_accuracies = []
for epoch in range(1, 201):
    loss = train()  # 假设train()函数返回当前epoch的损失
    train_losses.append(loss.item())  # 记录损失
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc = test(data.test_mask)  # 假设test()函数返回当前epoch的测试准确率
    test_accuracies.append(test_acc)  # 记录测试准确率
    print(f'Test Accuracy: {test_acc:.4f}')

# 创建图和轴对象
fig, ax1 = plt.subplots(figsize=(10, 5))
# 绘制训练损失曲线
ax1.plot(train_losses, label='Training Loss', color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
# 创建第二个y轴
ax2 = ax1.twinx()
# 绘制测试准确率曲线
ax2.plot(test_accuracies, label='Test Accuracy', color='red')
ax2.set_ylabel('Test Accuracy', color='red')
ax2.tick_params(axis='y', labelcolor='red')
# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# 添加标题
plt.title('Training Loss and Test Accuracy')
# 显示图表
plt.show()

# =============================================================================
# 可视化训练模型的输出嵌入
model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)
