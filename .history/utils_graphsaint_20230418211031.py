import scipy.sparse as sp
import numpy as np
import sys
import json
import os
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
import torch
from itertools import repeat
from torch_geometric.data import NeighborSampler

class DataGraphSAINT:
    '''GraphSAINT论文中使用的数据集'''

    def __init__(self, dataset, **kwargs):
        dataset_str='data/'+dataset+'/'
        adj_full = sp.load_npz(dataset_str+'adj_full.npz') # 加载原始邻接矩阵
        self.nnodes = adj_full.shape[0] # 节点数量

        if dataset == 'ogbn-arxiv': # 对于ogbn-arxiv数据集，进行对称归一化处理
            adj_full = adj_full + adj_full.T
            adj_full[adj_full > 1] = 1

        role = json.load(open(dataset_str+'role.json','r')) # 加载节点角色信息
        idx_train = role['tr'] # 训练集节点索引
        idx_test = role['te'] # 测试集节点索引
        idx_val = role['va'] # 验证集节点索引

        if 'label_rate' in kwargs: # 如果传入了'label_rate'参数，根据比例裁剪训练集节点
            label_rate = kwargs['label_rate']
            if label_rate < 1:
                idx_train = idx_train[:int(label_rate*len(idx_train))]

        self.adj_train = adj_full[np.ix_(idx_train, idx_train)] # 训练集邻接矩阵
        self.adj_val = adj_full[np.ix_(idx_val, idx_val)] # 验证集邻接矩阵
        self.adj_test = adj_full[np.ix_(idx_test, idx_test)] # 测试集邻接矩阵

        feat = np.load(dataset_str+'feats.npy') # 加载节点特征
        # ---- 标准化节点特征 ----
        feat_train = feat[idx_train] # 训练集节点特征
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)

        self.feat_train = feat[idx_train] # 训练集节点特征
        self.feat_val = feat[idx_val] # 验证集节点特征
        self.feat_test = feat[idx_test] # 测试集节点特征

        class_map = json.load(open(dataset_str + 'class_map.json','r')) # 加载节点类别映射信息
        labels = self.process_labels(class_map) # 处理节点类别信息，得到标签数组

        self.labels_train = labels[idx_train] # 训练集节点标签
        self.labels_val = labels[idx_val] # 验证集节点标签
        self.labels_test = labels[idx_test] # 测试集节点标签

        self.data_full = GraphData(adj_full, feat, labels, idx_train, idx_val, idx_test) # 构建完整的图数据
        self.class_dict = None
        self.class_dict2 = None

        self.adj_full = adj_full # 完整的邻接矩阵
        self.feat_full = feat # 完整的节点特征
        self.labels_full = labels # 完整的节点标签
        self.idx_train = np.array(idx_train) # 训练集节点索引
        self.idx_val = np.array(idx_val) # 验证集节点索引
        self.idx_test = np.array(idx_test)# 测试集节点索引
        self.samplers = None

    def process_labels(self, class_map):
        """
        为输出类别设置顶点属性映射
        """
        num_vertices = self.nnodes  # 获取顶点数量
        if isinstance(list(class_map.values())[0], list):  # 判断类别映射是否为列表
            num_classes = len(list(class_map.values())[0])  # 获取类别数量
            self.nclass = num_classes  # 设置类别数量属性
            class_arr = np.zeros((num_vertices, num_classes))  # 创建用于存储类别的数组
            for k,v in class_map.items():
                class_arr[int(k)] = v  # 将类别映射值存入数组中
        else:
            class_arr = np.zeros(num_vertices, dtype=np.int)  # 创建用于存储类别的数组
            for k, v in class_map.items():
                class_arr[int(k)] = v  # 将类别映射值存入数组中
            class_arr = class_arr - class_arr.min()  # 将类别数组的最小值设为0
            self.nclass = max(class_arr) + 1  # 设置类别数量属性
        return class_arr  # 返回类别数组

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}  # 创建类别字典
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)  # 将类别映射为布尔数组并存入字典中
        idx = np.arange(len(self.labels_train))  # 创建包含所有训练标签索引的数组
        idx = idx[self.class_dict['class_%s'%c]]  # 获取特定类别的索引
        return np.random.permutation(idx)[:num]  # 随机排列索引并返回指定数量的索引


    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        # 根据输入参数获取采样器的采样大小
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]

        # 如果class_dict2为空，则根据标签和训练集索引创建类别字典
        if self.class_dict2 is None:
            print(sizes)
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        # 如果samplers为空，则创建采样器
        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    continue

                self.samplers.append(NeighborSampler(adj,
                                                     node_idx=node_idx,
                                                     sizes=sizes, batch_size=num,
                                                     num_workers=8, return_e_id=False,
                                                     num_nodes=adj.size(0),
                                                     shuffle=True))
        # 从指定类别的类别字典中随机选择一批样本进行采样
        batch = np.random.permutation(self.class_dict2[c])[:num]
        # 使用采样器对选择的样本进行采样
        out = self.samplers[c].sample(batch)
        return out

class GraphData:
    """
    图数据类，用于存储图数据的相关信息，包括邻接矩阵、特征矩阵、标签、训练集索引、验证集索引和测试集索引。
    """

    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test):

        """
        初始化函数,用于创建GraphData对象。

        参数:
            adj (scipy.sparse.csr_matrix): 图的邻接矩阵，以稀疏矩阵的形式表示。
            features (numpy.ndarray): 节点的特征矩阵,以numpy数组的形式表示。
            labels (numpy.ndarray): 节点的标签,以numpy数组的形式表示。
            idx_train (numpy.ndarray): 训练集节点的索引,以numpy数组的形式表示。
            idx_val (numpy.ndarray): 验证集节点的索引,以numpy数组的形式表示。
            idx_test (numpy.ndarray): 测试集节点的索引,以numpy数组的形式表示。
        """
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test


class Data2Pyg:
    def __init__(self, data, device='cuda', transform=None, **kwargs):
        """
        初始化函数,用于初始化Data2Pyg类的实例对象。

        参数:
            data: 一个包含数据集的对象，应包含以下属性：
                  - data_train: 训练集数据
                  - data_val: 验证集数据
                  - data_test: 测试集数据
                  - nclass: 类别数
                  - nfeat: 特征数
            device: str, optional, 默认为'cuda'
                数据处理设备，可以是'cuda'或'cpu'。
            transform: object, optional, 默认为None
                数据预处理的可调用对象，用于对数据集进行变换。
            **kwargs: dict
                其他参数。

        返回:
            无返回值。
        """
        self.data_train = Dpr2Pyg(data.data_train, transform=transform)[0].to(device) # 将训练集数据转换为PyG数据格式，并存储在self.data_train中
        self.data_val = Dpr2Pyg(data.data_val, transform=transform)[0].to(device) # 将验证集数据转换为PyG数据格式，并存储在self.data_val中
        self.data_test = Dpr2Pyg(data.data_test, transform=transform)[0].to(device) # 将测试集数据转换为PyG数据格式，并存储在self.data_test中
        self.nclass = data.nclass # 存储类别数到self.nclass
        self.nfeat = data.nfeat # 存储特征数到self.nfeat
        self.class_dict = None # 初始化类别字典为None

    def retrieve_class(self, c, num=256):
        """
        检索指定类别的样本索引。

        参数:
            c: int
                指定的类别。
            num: int, optional, 默认为256
                随机采样的样本数量。

        返回:
            idx: numpy.array
                随机采样的指定类别的样本索引。
        """
        if self.class_dict is None: # 如果类别字典为空，则进行初始化
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.data_train.y == i).cpu().numpy() # 将训练集中属于类别i的样本标记为True，其余样本标记为False，并存储在类别字典中
        idx = np.arange(len(self.data_train.y)) # 生成训练集样本的索引
        idx = idx[self.class_dict['class_%s'%c]] # 获取指定类别c的样本索引
        return np.random.permutation(idx)[:num] # 随机打乱样本索引并返回指定数量的索引



class Dpr2Pyg(InMemoryDataset):
    """
    将DPR数据转换为PyTorch Geometric (PyG)数据集的类。

    Args:
        dpr_data (object): 包含DPR数据的对象。
        transform (callable, optional): 数据转换函数。默认为None。
        **kwargs: 传递给父类InMemoryDataset的参数。

    Attributes:
        root (str): 数据集的根目录。这里使用了虚拟的路径'data/'，没有实际意义。
        dpr_data (object): 包含DPR数据的对象。
        data (torch_geometric.data.Data): 存储转换后的PyG数据集。
        slices (dict): 数据集的切片字典，用于从data中提取子图。
        transform (callable): 数据转换函数。

    """

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/' # 虚拟的根目录路径，没有实际意义
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process(self):
        """
        将DPR数据转换为PyG数据。

        Returns:
            torch_geometric.data.Data: 转换后的PyG数据。

        """
        dpr_data = self.dpr_data
        edge_index = torch.LongTensor(dpr_data.adj.nonzero())  # 从DPR数据的邻接矩阵中提取边索引
        if sp.issparse(dpr_data.features):  # 判断特征是否为稀疏矩阵
            x = torch.FloatTensor(dpr_data.features.todense()).float()  # 如果是稀疏矩阵，则转换为密集矩阵
        else:
            x = torch.FloatTensor(dpr_data.features).float()  # 否则，将特征转换为PyTorch张量
        y = torch.LongTensor(dpr_data.labels)  # 将标签转换为PyTorch张量
        data = Data(x=x, edge_index=edge_index, y=y)  # 创建PyG数据对象
        data.train_mask = None  # 训练掩码，默认为None
        data.val_mask = None  # 验证掩码，默认为None
        data.test_mask = None  # 测试掩码，默认为None
        return data


    def get(self, idx):
        """
        获取指定索引的数据项

        参数：
            - idx (int): 数据项的索引

        返回：
            - data (torch.Tensor): 数据项的副本

        """
        # 创建一个与self.data相同类别的新对象
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            # 如果self.data具有'__num_nodes__'属性，则将num_nodes属性设置为指定索引处的值
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            # 对于self.data中的每个关键字
            item, slices = self.data[key], self.slices[key]
            # 获取数据项和切片
            s = list(repeat(slice(None), item.dim()))
            # 创建一个切片列表，维度与item相同，每个维度的切片都设置为None
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            # 将当前关键字的切片替换为从slices中获取的切片
            data[key] = item[s]
            # 将切片应用到item并赋值给data的相应关键字

        return data

    @property
    def raw_file_names(self):
        """
        返回原始文件的文件名列表

        返回：
            - raw_file_names (list): 原始文件的文件名列表

        """
        return ['some_file_1', 'some_file_2', ...]  # 返回原始文件的文件名列表

    @property
    def processed_file_names(self):
        """
        返回已处理文件的文件名列表

        返回：
            - processed_file_names (list): 已处理文件的文件名列表

        """
        return ['data.pt']  # 返回已处理文件的文件名列表

    def _download(self):
        """
        下载数据的私有方法

        """
        pass  # 下载数据的私有方法，此处为空实现


