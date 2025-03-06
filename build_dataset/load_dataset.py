import os

import numpy as np
import torch
from torch_geometric.data import Dataset, Data

# 将16进制字符转换为独热码
def hex_to_one_hot(hex_char):
    one_hot = np.zeros(16)
    one_hot[int(hex_char, 16)] = 1
    return one_hot

# 将16进制字符串转换为独热码向量
def hex_string_to_one_hot_vector(hex_string):
    one_hot_vector = np.concatenate([hex_to_one_hot(char) for char in hex_string])
    return one_hot_vector

# 定义自己的数据集类
class mydataset(Dataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(mydataset, self).__init__(root, name, transform, pre_transform)

    # 原始文件位置
    @property
    def raw_file_names(self):
        names = ['x', 'edge_index', 'edge_feature']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self):
        return f'{self.name}_data.pt'

    @property
    def num_classes(self) -> int:
        return int(self.get(0).y.max().item() + 1)

    def download(self):
        pass

    def process(self):
        print('Processing...')
        dtype = [('id', int), ('name','U24'), ('level', float), ('score', float), ('label', int)]
        pre_datas = np.genfromtxt(self.raw_paths[0], dtype)
        pre_x = pre_datas['name']

        # x = idx_features_labels[:, 1:-1]
        x = np.array([hex_string_to_one_hot_vector(hex_string) for hex_string in pre_x])
        x = torch.tensor(x, dtype=torch.float32)
        y = pre_datas['label']
        y = torch.tensor(y, dtype=torch.int32)
        idx = np.array(pre_datas['id'], dtype=np.int32)
        id_node = {j: i for i, j in enumerate(idx)}

        edges_unordered = np.genfromtxt(self.raw_paths[1], dtype=np.int32)
        edge_str = [id_node[each[0]] for each in edges_unordered]
        edge_end = [id_node[each[1]] for each in edges_unordered]
        edge_index = torch.tensor([edge_str, edge_end], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)

        torch.save(data, os.path.join(self.processed_dir, f'{self.name}_data.pt'))

    def len(self):
        dtype = [('id', int), ('name', 'U24'), ('level', float), ('score', float), ('label', int)]
        pre_datas = np.genfromtxt(self.raw_paths[0], dtype)
        uid = pre_datas['id']
        return len(uid)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.name}_data.pt'), weights_only=False)
        return data

    # def __repr__(self) -> str:
    #     return f'{self.name}({len(self)})'






