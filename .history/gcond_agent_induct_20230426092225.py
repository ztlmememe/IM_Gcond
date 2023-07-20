import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
import scipy.sparse as sp
from torch_sparse import SparseTensor

# 这是一个名为GCond的类，它有train和testwithval两个方法。
# 在初始化时，它会生成一个synthetic graph，然后使用Gradient Matching算法来训练GNN模型。
# 在train方法中，它会使用Gradient Matching算法来更新synthetic graph和GNN模型。
# 在testwithval方法中，它会使用训练好的GNN模型来预测测试集的标签，并返回测试集的准确率。

class GCond:
    # 初始化函数，接收数据、参数和设备信息
    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

        # 计算需要生成的synthetic graph的节点数
        n = int(len(data.idx_train) * args.reduction_rate)

        # 获取特征维度
        d = data.feat_train.shape[1]

        # 初始化synthetic graph的节点数和特征
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))

        # 初始化PGE模型
        self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)

        # 生成synthetic graph的标签
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)
        
        # 重置synthetic graph的特征
        self.reset_parameters()

        # 初始化特征优化器和PGE模型优化器
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        
        # 打印synthetic graph的节点数和特征维度
        print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    # 生成合成标签，即根据原始标签生成一些新的标签
    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train) # 计算data.labels_train中每个类别出现的次数
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1]) 
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}

        # 根据args.reduction_rate参数来计算每个类别需要生成的节点数，这个参数是一个缩减率，用于控制生成的节点数与原始节点数的比例。
        # 如果这个参数为 0.5，那么生成的节点数就是原始节点数的一半。
        for ix, (c, num) in enumerate(sorted_counter):
            # 如果是最后一个类别，那么直接计算需要生成的节点数，并将这个类别的合成标签添加到 labels_syn 列表中
            if ix == len(sorted_counter) - 1: 
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                # 如果不是最后一个类别，那么先计算需要生成的节点数，然后将这个类别的合成标签添加到 labels_syn 列表中。
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1) 
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def test_with_val(self, verbose=True):
        res = []

        data, device = self.data, self.device
        # 通过 detach() 方法将其从计算图中分离出来，使其不再参与反向传播的计算，从而避免了梯度累加的问题。
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                self.pge, self.labels_syn
        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0

        # 定义了一个GCN模型，包括输入特征的维度、隐藏层的维度、dropout率、权重衰减、层数和输出类别数
        model = GCN(nfeat=feat_syn.shape[1], 
                    nhid=self.args.hidden, 
                    dropout=dropout,
                    weight_decay=5e-4, 
                    nlayers=2,
                    nclass=data.nclass, 
                    device=device).to(device)

        adj_syn = pge.inference(feat_syn) # 通过PGE进行推理，得到邻接矩阵
        args = self.args

        if args.save: # 如果需要保存生成图
            torch.save(adj_syn, f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            torch.save(feat_syn, f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        noval = True # 不使用验证集
        # 在训练集上训练模型
        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=600, normalize=True, verbose=False, noval=noval)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        output = model.predict(data.feat_test, data.adj_test)

        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)

        res.append(acc_test.item()) # 将测试集准确率添加到结果列表中
        # 打印测试集结果
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        print(adj_syn.sum(), adj_syn.sum()/(adj_syn.shape[0]**2))

        if False: # 如果需要在训练集上进行预测
            if self.args.dataset == 'ogbn-arxiv':
                thresh = 0.6
            elif self.args.dataset == 'reddit':
                thresh = 0.91
            else:
                thresh = 0.7

            labels_train = torch.LongTensor(data.labels_train).cuda()
            output = model.predict(data.feat_train, data.adj_train)
            # loss_train = F.nll_loss(output, labels_train)
            # acc_train = utils.accuracy(output, labels_train)
            loss_train = torch.tensor(0)
            acc_train = torch.tensor(0)
            if verbose:
                print("Train set results:",
                      "loss= {:.4f}".format(loss_train.item()),
                      "accuracy= {:.4f}".format(acc_train.item()))
            res.append(acc_train.item())
        return res

    def train(self, verbose=True):
        # 获取参数和数据
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        syn_class_indices = self.syn_class_indices

        # 将数据转换为张量并将其放置在设备上
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        # 获取子邻接矩阵和子特征矩阵
        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        # 将子特征矩阵复制到feat_syn中
        self.feat_syn.data.copy_(feat_sub)

        # 根据是否是稀疏张量来规范化邻接矩阵
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        # 将邻接矩阵转换为稀疏张量
        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()

        # 获取外部循环和内部循环的次数
        outer_loop, inner_loop = get_loops(args)

        for it in range(args.epochs+1):
            loss_avg = 0
            if args.sgc==1:
                model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)
            elif args.sgc==2:
                model = SGC1(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)

            else:
                model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                            device=self.device).to(self.device)

            model.initialize()

            model_parameters = list(model.parameters())

            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            # 进行外部循环
            for ol in range(outer_loop):
                # 生成并规范化合成邻接矩阵
                adj_syn = pge(self.feat_syn)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                feat_syn_norm = feat_syn

                # 判断是否有BatchNorm层
                BN_flag = False
                for module in model.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True

                # 如果有BatchNorm层，则需要训练模型以更新BatchNorm层的mu和sigma
                if BN_flag:
                    model.train() # for updating the mu, sigma of BatchNorm
                    output_real = model.forward(features, adj_norm)
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                loss = torch.tensor(0.0).to(self.device)
                for c in range(data.nclass):
                    if c not in self.num_class_dict:
                        continue
                    # 获取类别c的样本
                    batch_size, n_id, adjs = data.retrieve_class_sampler(
                            c, adj, transductive=False, args=args)

                    # 如果只有一层，则将邻接矩阵放入列表中
                    if args.nlayers == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]

                    # 根据采样结果进行前向传播，LOSS计算和梯度计算
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    # 获取类别c的合成邻接矩阵
                    ind = syn_class_indices[c]
                    if args.nlayers == 1:
                        adj_syn_norm_list = [adj_syn_norm[ind[0]: ind[1]]]
                    else:
                        adj_syn_norm_list = [adj_syn_norm]*(args.nlayers-1) + \
                                [adj_syn_norm[ind[0]: ind[1]]]

                    # 计算合成邻接矩阵下的输出和LOSS
                    output_syn = model.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_syn = F.nll_loss(output_syn, labels_syn[ind[0]: ind[1]])
                    # 计算合成邻接矩阵下的梯度
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                   
                    # 计算匹配损失
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)

                loss_avg += loss.item()
                # 计算正则化损失
                # TODO: regularize
                if args.alpha > 0:
                    loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(labels_syn))
                # else:
                else:
                    loss_reg = torch.tensor(0)

                loss = loss + loss_reg

                # 更新合成图
                self.optimizer_feat.zero_grad()
                # 每次计算梯度时，梯度都会被累加到梯度缓存中。
                # 因此，在每次更新模型参数之前需要将梯度缓存清零，以避免梯度累加的影响。
                self.optimizer_pge.zero_grad()
                loss.backward() # 计算损失函数对于模型参数的梯度

                # 根据 it 的值选择更新 self.optimizer_pge 或 self.optimizer_feat
                if it % 50 < 10:
                    self.optimizer_pge.step() # 使用优化算法来更新模型参数
                else:
                    self.optimizer_feat.step()

                if args.debug and ol % 5 ==0: # 打印梯度匹配损失
                    print('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break

                # 进行内循环，更新 GNN 模型的参数
                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step() # update gnn param

            # 计算平均损失并打印
            loss_avg /= (data.nclass*outer_loop)
            if it % 50 == 0:
                print('Epoch {}, loss_avg: {}'.format(it, loss_avg))

            eval_epochs = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000]

            if verbose and it in eval_epochs:
            # if verbose and (it+1) % 500 == 0:
                res = []
                runs = 1 if args.dataset in ['ogbn-arxiv', 'reddit', 'flickr'] else 3
                for i in range(runs):
                    # self.test()
                    res.append(self.test_with_val())
                res = np.array(res)
                print('Test:',
                        repr([res.mean(0), res.std(0)]))



    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        # 计算了 self.labels_syn.cpu().numpy() 中每个元素的出现次数
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[idx_selected]

        # adj_knn = torch.zeros((data.nclass*args.nsamples, data.nclass*args.nsamples)).to(self.device)
        # for i in range(data.nclass):
        #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
        #     adj_knn[np.ix_(idx, idx)] = 1

        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        return 10, 0

    if args.dataset in ['ogbn-arxiv']:
        return 20, 0
    if args.dataset in ['reddit']:
        return args.outer, args.inner
    if args.dataset in ['flickr']:
        return args.outer, args.inner
        # return 10, 1
    if args.dataset in ['cora']:
        return 20, 10
    if args.dataset in ['citeseer']:
        return 20, 5 # at least 200 epochs
    else:
        return 20, 5

