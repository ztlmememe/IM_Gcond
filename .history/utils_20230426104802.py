import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test
from torch_geometric.utils import train_test_split_edges
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from deeprobust.graph.utils import *
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_geometric.datasets import Planetoid


def get_dataset(name, normalize_features=False, transform=None, if_dpr=True):
    """
    根据数据集名称获取数据集，并进行相应的预处理操作。

    参数:
        - name (str): 数据集名称，可选值为 'cora', 'citeseer', 'pubmed', 'ogbn-arxiv'。
        - normalize_features (bool): 是否对特征进行归一化，默认为 False。
        - transform (torchvision.transforms.Compose): 数据集预处理的转换操作，默认为 None。
        - if_dpr (bool): 是否进行 DPR 格式的处理，默认为 True。

    返回:
        - dpr_data (Pyg2Dpr): 处理后的数据集对象。

    Raises:
        - NotImplementedError: 如果输入的数据集名称不在支持的列表中，则抛出 NotImplementedError 异常。
    """
    # 拼接数据集路径
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    # 根据数据集名称选择相应的处理方式
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    else:
        raise NotImplementedError

    # 根据参数进行数据集预处理操作
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    # 进行 DPR 格式的处理
    dpr_data = Pyg2Dpr(dataset)
    if name in ['ogbn-arxiv']:
        # ogbn-arxiv 数据集的特征需要进行归一化处理，遵循 GraphSAINT 的方式
        feat, idx_train = dpr_data.features, dpr_data.idx_train
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)
        dpr_data.features = feat

    return dpr_data



class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, **kwargs):
        # 初始化函数，接受参数 pyg_data 和 kwargs

        try:
            splits = pyg_data.get_idx_split() # 尝试从 pyg_data 获取索引划分信息
        except:
            pass

        dataset_name = pyg_data.name # 获取数据集名称
        pyg_data = pyg_data[0] # 获取 pyg_data 的第一个元素
        n = pyg_data.num_nodes # 获取节点数量

        if dataset_name == 'ogbn-arxiv': # 如果数据集名称为 'ogbn-arxiv'，进行对称化处理
            pyg_data.edge_index = to_undirected(pyg_data.edge_index, pyg_data.num_nodes)

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]), # 创建一个稀疏矩阵，用于存储边的连接关系
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))

        self.features = pyg_data.x.numpy() # 将节点特征转换为 numpy 数组并赋值给 self.features
        self.labels = pyg_data.y.numpy() # 将节点标签转换为 numpy 数组并赋值给 self.labels

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # 对于 ogb-arxiv 数据集，需要对标签进行形状变换

        if hasattr(pyg_data, 'train_mask'): # 如果 pyg_data 具有 'train_mask' 属性
            # 对于固定的划分
            self.idx_train = mask_to_index(pyg_data.train_mask, n) # 调用 mask_to_index 函数将训练集掩码转换为索引
            self.idx_val = mask_to_index(pyg_data.val_mask, n) # 调用 mask_to_index 函数将验证集掩码转换为索引
            self.idx_test = mask_to_index(pyg_data.test_mask, n) # 调用 mask_to_index 函数将测试集掩码转换为索引
            self.name = 'Pyg2Dpr' # 设置数据集名称为 'Pyg2Dpr'
        else:
            try:
                # 对于 ogb 数据集
                self.idx_train = splits['train'] # 从索引划分信息中获取训练集索引
                self.idx_val = splits['valid'] # 从索引划分信息中获取验证集索引
                self.idx_test = splits['test'] # 从索引划分信息中获取测试集索引
                self.name = 'Pyg2Dpr' # 设置数据集名称为 'Pyg2Dpr'
            except:
                # 对于其他数据集
                self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                        nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels) # 调用 get_train_val_test 函数生成训练集、验证集、测试集的索引




def mask_to_index(index, size):
    """
    将给定的索引转换为对应的全索引数组。
    
    参数：
    index(int或ndarray):需要转换的索引值或索引数组。
    size(int):全索引数组的大小。
    
    返回：
    all_idx(ndarray)转换后的全索引数组。
    """
    all_idx = np.arange(size)  # 生成从0到size-1的全索引数组
    return all_idx[index]  # 返回对应的全索引数组中的值


def index_to_mask(index, size):
    """
    将给定的索引转换为对应的掩码数组。
    
    参数：
    index(int或ndarray):需要转换的索引值或索引数组。
    size(int):全索引数组的大小。
    
    返回：
    mask(torch.Tensor):转换后的掩码数组,类型为torch.bool。
    """
    mask = torch.zeros((size, ), dtype=torch.bool)  # 生成全零的掩码数组
    mask[index] = 1  # 将指定索引位置设为1，表示对应位置为True
    return mask  # 返回掩码数组



class Transd2Ind:
    # transductive setting to inductive setting

    def __init__(self, dpr_data, keep_ratio):
        """
        初始化函数,用于将传入的数据从半监督学习(transductive setting)转换为有监督学习(inductive setting)。

        参数:
            - dpr_data: 数据集对象，包含了索引、邻接矩阵、特征和标签等数据。
            - keep_ratio: 保留的样本比例，用于从训练集中随机选择一部分样本用于有监督学习。

        """
        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        adj, features, labels = dpr_data.adj, dpr_data.features, dpr_data.labels
        self.nclass = labels.max()+1
        self.adj_full, self.feat_full, self.labels_full = adj, features, labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)

        if keep_ratio < 1:
            idx_train, _ = train_test_split(idx_train,
                                            random_state=None,
                                            train_size=keep_ratio,
                                            test_size=1-keep_ratio,
                                            stratify=labels[idx_train])

        self.adj_train = adj[np.ix_(idx_train, idx_train)]
        self.adj_val = adj[np.ix_(idx_val, idx_val)]
        self.adj_test = adj[np.ix_(idx_test, idx_test)]
        print('size of adj_train:', self.adj_train.shape)
        print('#edges in adj_train:', self.adj_train.sum())

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]

        self.feat_train = features[idx_train]
        self.feat_val = features[idx_val]
        self.feat_test = features[idx_test]

        self.class_dict = None
        self.samplers = None
        self.class_dict2 = None

    def retrieve_class(self, c, num=256):
        """
        根据指定的类别c,随机选择指定数量的训练样本进行采样。

        参数:
            - c: 类别标签,从0到self.nclass-1。
            - num: 采样的样本数量,默认为256。

        返回:
            - 采样得到的训练样本的索引数组。

        """
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        """
        根据类别c从邻居采样器中检索采样数据。

        参数:
            - c (int): 类别索引
            - adj (torch.Tensor): 邻接矩阵
            - transductive (bool): 是否使用跨领域设置
            - num (int): 采样数量
            - args (argparse.Namespace): 命令行参数

        返回:
            - out (torch.Tensor): 采样的数据
        """
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=12, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        # out = self.samplers[c].sample(batch)
        out = self.samplers[c].sample(torch.from_numpy(batch).long())
        return out


    def retrieve_class_multi_sampler(self, c, adj, transductive, num=256, args=None):
        """
        根据类别c从多层邻居采样器中检索采样数据。

        参数:
            - c (int): 类别索引
            - adj (torch.Tensor): 邻接矩阵
            - transductive (bool): 是否使用跨领域设置
            - num (int): 采样数量
            - args (argparse.Namespace): 命令行参数

        返回:
            - out (torch.Tensor): 采样的数据
        """
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx


        if self.samplers is None:
            self.samplers = []
            for l in range(2):
                layer_samplers = []
                sizes = [15] if l == 0 else [10, 5]
                for i in range(self.nclass):
                    node_idx = torch.LongTensor(self.class_dict2[i])
                    layer_samplers.append(NeighborSampler(adj,
                                        node_idx=node_idx,
                                        sizes=sizes, batch_size=num,
                                        num_workers=12, return_e_id=False,
                                        num_nodes=adj.size(0),
                                        shuffle=True))
                self.samplers.append(layer_samplers)
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[args.nlayers-1][c].sample(batch)
        return out



def match_loss(gw_syn, gw_real, args, device):
    """
    计算生成图像和真实图像之间的匹配损失，用于衡量生成图像与真实图像的相似度

    参数:
        - gw_syn (list): 生成图像的特征向量列表
        - gw_real (list): 真实图像的特征向量列表
        - args (argparse.Namespace): 命令行参数对象
        - device (torch.device): 设备对象，用于将计算放在指定的设备上

    返回:
        - dis (torch.Tensor): 匹配损失，作为生成图像和真实图像之间的相似度指标
    """
    dis = torch.tensor(0.0).to(device)  # 初始化匹配损失

    if args.dis_metric == 'ours':  # 使用自定义的距离度量函数

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':  # 使用均方误差作为距离度量函数
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':  # 使用余弦相似度作为距离度量函数
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')  # 如果距离度量函数未知，则抛出错误

    return dis

def distance_wb(gwr, gws):
    """
    计算权重参数之间的距离，用于权重剪枝。

    参数:
    - gwr: torch.Tensor, 权重参数的一个副本
    - gws: torch.Tensor, 权重参数的一个副本

    返回:
    - dis: float, 权重参数之间的距离

    """
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def calc_f1(y_true, y_pred, is_sigmoid):
    """
    计算二分类或多分类任务的F1 score。

    参数:
    - y_true: np.ndarray, 真实标签
    - y_pred: np.ndarray, 预测标签
    - is_sigmoid: bool, 是否使用sigmoid激活函数

    返回:
    - micro: float, micro-average F1 score
    - macro: float, macro-average F1 score

    """
    if not is_sigmoid:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def evaluate(output, labels, args):
    """
    评估模型在测试集上的性能。

    参数:
    - output: torch.Tensor, 模型在测试集上的输出
    - labels: torch.Tensor, 测试集的真实标签
    - args: argparse.Namespace, 命令行参数对象

    返回:
    无

    """
    data_graphsaint = ['yelp', 'ppi', 'ppi-large', 'flickr', 'reddit', 'amazon']
    if args.dataset in data_graphsaint:
        labels = labels.cpu().numpy()
        output = output.cpu().numpy()
        if len(labels.shape) > 1:
            micro, macro = calc_f1(labels, output, is_sigmoid=True)
        else:
            micro, macro = calc_f1(labels, output, is_sigmoid=False)
        print("Test set results:", "F1-micro= {:.4f}".format(micro),
                "F1-macro= {:.4f}".format(macro))
    else:
        loss_test = F.nll_loss(output, labels)
        acc_test = accuracy(output, labels)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    return


from torchvision import datasets, transforms
def get_mnist(data_path):
    """
    从MNIST数据集中获取图像数据,并进行数据预处理和标准化

    参数:
        - data_path (str): MNIST数据集路径

    返回:
        - Dpr2Pyg: 经过预处理和标准化后的MNIST数据集
    """
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]

    # 定义数据预处理和标准化的变换
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    # 加载MNIST数据集的训练集和测试集
    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no        augmentation
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]

    labels = []
    feat = []
    # 遍历训练集，提取图像特征和标签
    for x, y in dst_train:
        feat.append(x.view(1, -1))
        labels.append(y)
    feat = torch.cat(feat, axis=0).numpy()

    from utils_graphsaint import GraphData
    # 构建GraphData对象
    adj = sp.eye(len(feat))
    idx = np.arange(len(feat))
    dpr_data = GraphData(adj-adj, feat, labels, idx, idx, idx)

    from deeprobust.graph.data import Dpr2Pyg
    # 转换为Dpr2Pyg对象并返回
    return Dpr2Pyg(dpr_data)

def regularization(adj, x, eig_real=None):
    """
    计算正则化项的损失函数

    参数:
        - adj (torch.Tensor): 图的邻接矩阵
        - x (torch.Tensor): 图的特征矩阵
        - eig_real (None): 未使用

    返回:
        - torch.Tensor: 正则化项的损失值
    """
    # fLf
    loss = 0
    # loss += torch.norm(adj, p=1)
    loss += feature_smoothing(adj, x)
    return loss

def maxdegree(adj):
    """
    计算邻接矩阵的最大度数

    参数:
        - adj (torch.Tensor): 图的邻接矩阵

    返回:
        - torch.Tensor: 最大度数的值
    """

    n = adj.shape[0]
    return F.relu(max(adj.sum(1))/n - 0.5)

def sparsity2(adj):
    """
    计算邻接矩阵的稀疏性损失函数

    参数:
        - adj (torch.Tensor): 图的邻接矩阵

    返回:
        - torch.Tensor: 稀疏性损失值
    """
    n = adj.shape[0]
    loss_degree = - torch.log(adj.sum(1)).sum() / n
    loss_fro = torch.norm(adj) / n
    return 0 * loss_degree + loss_fro

def sparsity(adj):
    """
    计算邻接矩阵的稀疏度。

    参数：
    adj (torch.Tensor): 输入的邻接矩阵，形状为 (n, n)。

    返回：
    torch.Tensor: 稀疏度，通过对邻接矩阵求和后减去阈值并应用 ReLU 激活函数得到。
    """
    n = adj.shape[0]  # 获取邻接矩阵的维度
    thresh = n * n * 0.01  # 设置稀疏度的阈值
    return F.relu(adj.sum() - thresh)  # 计算稀疏度并应用 ReLU 激活函数
    # 可选：返回稀疏度除以邻接矩阵大小的平方
    # return F.relu(adj.sum()-thresh) / n**2


def feature_smoothing(adj, X):
    """
    计算特征平滑损失。

    参数：
    adj (torch.Tensor): 输入的邻接矩阵，形状为 (n, n)。
    X (torch.Tensor): 输入的特征矩阵，形状为 (n, d)，其中 n 是节点数,d 是特征维度。

    返回：
    torch.Tensor: 特征平滑损失，通过对特征矩阵和邻接矩阵进行矩阵运算得到。
    """
    adj = (adj.t() + adj) / 2  # 对称化邻接矩阵
    rowsum = adj.sum(1)  # 计算每行邻接矩阵的和
    r_inv = rowsum.flatten()  # 将每行和转换为一维向量
    D = torch.diag(r_inv)  # 构建度矩阵

    r_inv = r_inv + 1e-8  # 添加一个小的常数，防止除零错误
    r_inv = r_inv.pow(-1/2).flatten()  # 计算度矩阵的逆平方根并转换为一维向量
    r_inv[torch.isinf(r_inv)] = 0.  # 将无穷大的值替换为零，避免 NaN
    r_mat_inv = torch.diag(r_inv)  # 构建度矩阵逆平方根的对角矩阵

    L = r_mat_inv @ (D - adj) @ r_mat_inv  # 计算拉普拉斯矩阵

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)  # 计算特征平滑损失
    # 可选：将特征平滑损失除以邻接矩阵大小的平方
    # loss_smooth_feat = loss_smooth_feat / (adj.shape[0]**2)
    return torch.trace(XLXT)  # 返回特征平滑损失

def row_normalize_tensor(mx):
    """
    对输入的张量进行行归一化操作。

    参数：
    -- mx (torch.Tensor): 输入的待归一化的张量

    返回：
    -- torch.Tensor: 归一化后的张量
    """
    # 计算每行元素之和
    rowsum = mx.sum(1)
    # 计算每行元素之和的倒数，并展平成一维张量
    r_inv = rowsum.pow(-1).flatten()
    # 创建对角矩阵，其对角线上的元素为每行元素之和的倒数
    r_mat_inv = torch.diag(r_inv)
    # 将输入张量与对角矩阵相乘，实现行归一化
    mx = r_mat_inv @ mx
    # 返回归一化后的张量
    return mx


