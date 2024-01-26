import time
import argparse
import os
import gc
import random
import math
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from load_data import *
from encoder import *
from decoder import *
from utils import *

import logging
from torch.utils.tensorboard import SummaryWriter

class Experiment:
    def __init__(self, args):
        print("Experiment init")
        self.args = args

        self.data_dir = args.data_dir
        self.epoch = args.epoch
        self.train_batch_size = args.train_batch_size
        self.early_stop = args.early_stop
        self.skip_conn = args.skip_conn
        self.ent_init = args.ent_init
        self.activation = args.activation
        self.encoder = args.encoder.lower()
        self.hiddens = list(map(int, args.hiddens.split(",")))
        self.heads = list(map(int, args.heads.split(",")))
        self.decoder = args.decoder.lower()
        self.lr = args.lr

        self.early_stop_result = ()
        self.best_epoch = 0
        self.best_result = ()
        self.top_k = [1, 3, 5, 10]
        self.train_dist = 'euclidean'
        self.test_dist = 'euclidean'
        self.update = 5
        self.cached_sample = {}
        self.k = 25
        self.alpha = 1
        self.sampling = "N"
        self.alpha = 1
        self.margin = 1
        self.feat_drop = 0
        self.bias = True

    def init_embeddings(self):
        print("init_embeddings")
        # 创建实体和关系嵌入向量矩阵，并将其移动到设备上
        self.rel_embeddings = nn.Embedding(d.rel_num, self.hiddens[0]).to(device)
        nn.init.xavier_normal_(self.rel_embeddings.weight)
        if self.ent_init == "random":
            self.ent_embeddings = nn.Embedding(d.ent_num, self.hiddens[0]).to(device)
            nn.init.xavier_normal_(self.ent_embeddings.weight)
        elif self.ent_init == "name":
            with open(file= self.data_dir + '/' + self.data_dir[12:14] + '_vectorList.json', mode='r', encoding='utf-8') as f:
                embedding_list = json.load(f)
                print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
                input_embeddings = torch.tensor(embedding_list)  # 将 embedding_list 转换为 PyTorch 张量
                input_embeddings = F.normalize(input_embeddings, p=2, dim=1)
                self.ent_embeddings = nn.Embedding(
                    num_embeddings=input_embeddings.shape[0],  # 嵌入矩阵的行数
                    embedding_dim=input_embeddings.shape[1]  # 嵌入向量的维度
                )
                self.ent_embeddings.weight.data.copy_(input_embeddings)
        else:
            raise NotImplementedError("bad ent_init")
        
        # 初始化 encoder 之后的 enh_ent_embeddings 为 numpy 数组
        self.enh_ent_embeddings = self.ent_embeddings.weight.cpu().detach().numpy()

    def train_and_eval(self):
        print("train_and_eval")
        self.init_embeddings()

        # 初始化 encoder 和 decoder
        encoder = Encoder(name=self.encoder, hiddens=self.hiddens, heads=self.heads, skip_conn=self.skip_conn, activation=self.activation, feat_drop=self.feat_drop, bias=self.bias, ent_num=d.ent_num, rel_num=d.rel_num).to(device)
        logger.info(encoder)  # 输出encoder信息

        decoder = Decoder(name=self.decoder, hiddens=self.hiddens, skip_conn=self.skip_conn, train_dist=self.train_dist, sampling=self.sampling, alpha=self.alpha, margin=self.margin).to(device)
        logger.info(decoder)
        
        # 定义参数列表和优化器
        params = nn.ParameterList([self.ent_embeddings.weight, self.rel_embeddings.weight] + (list(decoder.parameters())) + (list(encoder.parameters())) )
        opt = optim.Adagrad(params, lr=self.lr)
        logger.info(params)
        logger.info(opt)
        
        # 训练
        logger.info("Start training...")
        for it in range(0, self.epoch): # 循环 epoch 次 

            if (len(d.ill_train_idx) == 0):  # 如果没有训练数据，则跳过该解码器的训练
                continue

            t_ = time.time()  # 记录开始时间

            # 训练一个 epoch,得到 loss,同时得到 self.enh_ent_embbeddings,即加强后的实体嵌入
            loss = self.train_1_epoch(it, opt, encoder, decoder, d.sparse_edges_idx, d.sparse_values, d.sparse_rels_idx, d.triple_idx, d.ill_train_idx, [d.kg1_ent_ids, d.kg2_ent_ids], self.ent_embeddings.weight, self.rel_embeddings.weight)
            writer.add_scalar("loss", loss, it)  # 在tensorboard中记录损失值
            logger.info("epoch %d: %.8f\ttime: %ds" % (it, loss, int(time.time()-t_)) )  # 输出当前迭代的训练结果

            # 先每一个 epoch val 一下
            logger.info("Start validating...")
            with torch.no_grad():  # 关闭梯度计算

                # 取 encoder 更新出的 embeddings
                embeddings = self.enh_ent_embeddings
                if len(d.ill_val_idx) > 0:  # 如果有验证集，则使用验证集进行评估
                    result = self.evaluate(it, d.ill_val_idx, embeddings)
                else:  # 否则使用测试集进行评估
                    result = self.evaluate(it, d.ill_test_idx, embeddings)

            # Early Stop
            # 如果开启了early stop并且当前结果比历史最佳结果差，则提前结束训练
            if self.early_stop and len(self.early_stop_result) != 0 and result[0][0] < self.early_stop_result[0][0]:  
                if len(d.ill_val_idx) > 0:
                    logger.info("Start testing...")
                    self.evaluate(it, d.ill_test_idx, embeddings)
                else:
                    logger.info("Early stop, best result:")
                    acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l = self.early_stop_result
                    logger.info("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f} ".format(self.top_k, acc_l2r, mean_l2r, mrr_l2r))
                    logger.info("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f} \n".format(self.top_k, acc_r2l, mean_r2l, mrr_r2l))
                break
            self.early_stop_result = result

            # 记录最好结果
            if  len(self.best_result) == 0 or result[0][0] > self.best_result[0][0]:  
                self.best_epoch = it
                self.best_result = result


    def train_1_epoch(self, it, opt, encoder, decoder, edges, values, rels, triples, ills, ids, ent_emb, rel_emb):
        print("train_1_epoch")
        #self.enh_ent_embbeddings = torch.tensor(self.ent_embeddings.weight)
        #return 1
        encoder.train()
        decoder.train()
        losses = []
        
        # 没太懂
        # 判断是否需要更新cached_sample（缓存的样本）
        if decoder.name not in self.cached_sample or it % self.update == 0:
            # 根据decoder的类型，设置pos_batch的值
            if decoder.name == "align":
                self.cached_sample[decoder.name] = ills.tolist()
                self.cached_sample[decoder.name] = np.array(self.cached_sample[decoder.name])
            else:
                self.cached_sample[decoder.name] = triples
            np.random.shuffle(self.cached_sample[decoder.name])
        
        # 获取训练样本
        train = self.cached_sample[decoder.name]

        # 设置训练批次大小
        if self.train_batch_size == -1:
            train_batch_size = len(train)
        else:
            train_batch_size = self.train_batch_size
        # 循环处理每个批次的样本
        for i in range(0, len(train), train_batch_size):
            # 获取正样本
            pos_batch = train[i:i+train_batch_size]

            # 判断是否需要更新cached_sample和进行采样
            if (decoder.name+str(i) not in self.cached_sample or it % self.update == 0) and decoder.sampling_method:
                #print("it len(pos_batch, triples, ills), len(ids[0]), decoder.k", it, len(pos_batch), len(triples), len(ills), decoder.k)
                self.cached_sample[decoder.name+str(i)] = decoder.sampling_method(pos_batch, triples, ills, ids, self.k, params={
                    "emb": self.enh_ent_embeddings,
                    "metric": self.test_dist,
                })


            # 获取负样本
            neg_batch = self.cached_sample[decoder.name+str(i)]

            # 清除梯度
            opt.zero_grad()

            # 构建长度相同的 neg 和 pos
            neg = torch.LongTensor(neg_batch).to(device)
            if neg.size(0) > len(pos_batch) * self.k:
                pos = torch.LongTensor(pos_batch).repeat(self.k * 2, 1).to(device)
            elif hasattr(decoder.func, "loss"):
                pos = torch.LongTensor(pos_batch).to(device)
            else:
                pos = torch.LongTensor(pos_batch).repeat(self.k, 1).to(device)

            # 获取增强嵌入 enh_emb ,即经过encoder的嵌入
            use_edges = torch.LongTensor(edges).to(device)
            use_rels = torch.LongTensor(rels).to(device)
            
            print("before encoder")
            if self.encoder == "gcn-align":
                enh_emb = encoder.forward(edges=use_edges, rels=None, x=ent_emb, r=rel_emb[d.sparse_rels_idx])
            elif self.encoder == "kecg":
                #kecg 有自环
                edges_with_self_loop = edges
                for i in range(d.ent_num):
                    edges_with_self_loop = np.concatenate((edges_with_self_loop, [[i, i]]))
                use_edges = torch.LongTensor(edges_with_self_loop).to(device)
                enh_emb = encoder.forward(edges=use_edges, rels=use_rels, x=ent_emb, r=rel_emb)
            else:
                enh_emb = encoder.forward(edges=use_edges, rels=use_rels, x=ent_emb, r=rel_emb)
            print("after encoder")
            # 更新增强实体嵌入 enh_ins_emb
            self.enh_ins_emb =  enh_emb.cpu().detach().numpy()
            print("self.enh_ins_emb ",self.enh_ins_emb)
            
            # 计算损失
            pos_score = decoder.forward(enh_emb, rel_emb, pos)
            neg_score = decoder.forward(enh_emb, rel_emb, neg)
            target = torch.ones(neg_score.size()).to(device)
            print("before decoder")
            loss = decoder.loss(pos_score, neg_score, target) * self.alpha
            print("after decoder")
            # 反向传播和参数更新
            loss.backward()
            opt.step()
            
            # 记录损失值
            losses.append(loss.item())
        
        # 返回平均损失
        return np.mean(losses)



    def evaluate(self, it, test, ent_emb):
        print("evaluate")
        # 记录评估开始时间
        t_test = time.time()
        # 指定用于计算 top-k 命中率的 k 值

        # 分别取验证/测试集中两个图的实体embedding
        left_emb = ent_emb[test[:, 0]]
        right_emb = ent_emb[test[:, 1]]

        # 计算左实体和右实体之间的欧几里得距离或余弦相似度，并按照距离从小到大排序
        # 先指定为欧几里得距离
        distance = - sim(left_emb, right_emb, metric=self.test_dist, normalize=True)

        # 将测试样本分成若干个子集，每个子集由一个进程来处理
        tasks = div_list(np.array(range(len(test))), 10)
        pool = multiprocessing.Pool(processes=len(tasks))
        reses = list()
        for task in tasks:
            # 并行计算 top-k 命中率和平均排名等指标
            reses.append(pool.apply_async(multi_cal_rank, (task, distance[task, :], distance[:, task], self.top_k, self.args)))
        pool.close()
        pool.join()
        
        # 合并各个子集的计算结果
        acc_l2r, acc_r2l = np.array([0.] * len(self.top_k)), np.array([0.] * len(self.top_k))
        mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0., 0., 0., 0.
        for res in reses:
            (_acc_l2r, _mean_l2r, _mrr_l2r, _acc_r2l, _mean_r2l, _mrr_r2l) = res.get()
            acc_l2r += _acc_l2r
            mean_l2r += _mean_l2r
            mrr_l2r += _mrr_l2r
            acc_r2l += _acc_r2l
            mean_r2l += _mean_r2l
            mrr_r2l += _mrr_r2l
        mean_l2r /= len(test)
        mean_r2l /= len(test)
        mrr_l2r /= len(test)
        mrr_r2l /= len(test)
        for i in range(len(self.top_k)):
            acc_l2r[i] = round(acc_l2r[i] / len(test), 4)
            acc_r2l[i] = round(acc_r2l[i] / len(test), 4)
        
        # 将计算结果记录到 TensorBoard 中
        logger.info("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(self.top_k, acc_l2r.tolist(), mean_l2r, mrr_l2r, time.time() - t_test))
        logger.info("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(self.top_k, acc_r2l.tolist(), mean_r2l, mrr_r2l, time.time() - t_test))
        for i, k in enumerate(self.top_k):
            writer.add_scalar("l2r_HitsAt{}".format(k), acc_l2r[i], it)
            writer.add_scalar("r2l_HitsAt{}".format(k), acc_r2l[i], it)
        writer.add_scalar("l2r_MeanRank", mean_l2r, it)
        writer.add_scalar("l2r_MeanReciprocalRank", mrr_l2r, it)
        writer.add_scalar("r2l_MeanRank", mean_r2l, it)
        writer.add_scalar("r2l_MeanReciprocalRank", mrr_r2l, it)

        # 返回计算结果
        return (acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/DBP15K/zh_en", required=False, help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--train_rate", type=float, default=0.3, help="training set rate")
    parser.add_argument("--val_rate", type=float, default=0.0, help="valid set rate")
    parser.add_argument("--epoch", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--train_batch_size", type=int, default=-1, help="train batch_size (-1 means all in)")
    parser.add_argument("--early_stop", action="store_true", default=False, help="whether to use early stop")
    
    parser.add_argument('--skip_conn', type=str, default='none', choices=['none', 'highway', 'concatall', 'concat0andl', 'residual'])
    parser.add_argument('--ent_init', type=str, default='random', choices=['random', 'name'])
    parser.add_argument('--activation', type=str, default='F.elu', choices=['none', 'F.elu'])

    parser.add_argument("--encoder", type=str, default="GCN-Align", nargs="?", help="which encoder to use: . max = 1")
    # 因为实体名称初始化的词典是300维的，所以默认用300的
    parser.add_argument("--hiddens", type=str, default="300,300,300", help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    
    # KECG 中使用
    parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
    
    parser.add_argument("--decoder", type=str, default="Align", nargs="?", help="which decoder to use: . min = 1")
    
    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")

    parser.add_argument("--log", type=str, default="tensorboard_log", nargs="?", help="where to save the log")
 
    args = parser.parse_args()  # 解析命令行参数

    logger = logging.getLogger(__name__)  # 创建日志记录器
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")  # 配置日志记录器的格式和级别
    writer = SummaryWriter("_runs%s/%s_%s" % (str(time.time()),args.data_dir.split("/")[-1], args.log))  # 创建tensorboard的SummaryWriter对象，用于记录训练过程和可视化
    logger.info(args)  # 将命令行参数打印到日志中

    args.seed = 12306
    args.cuda = False

    torch.backends.cudnn.deterministic = True  # 设置PyTorch的随机种子使其确定性
    random.seed(args.seed)  # 设置Python随机数生成器的种子
    np.random.seed(args.seed)  # 设置numpy随机数生成器的种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机数生成器的种子
    if args.cuda and torch.cuda.is_available():  # 如果cuda可用并且命令行参数指定使用cuda
        torch.cuda.manual_seed(args.seed)  # 设置PyTorch的cuda随机数生成器的种子
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")  # 根据cuda是否可用选择设备

    # 加载数据
    d = AlignmentData(data_dir=args.data_dir, train_rate=args.train_rate, val_rate=args.val_rate)
    logger.info(d)  # 将数据信息打印到日志中

    experiment = Experiment(args=args)  # 创建实验对象，传入命令行参数

    t_total = time.time()  # 记录开始时间
    experiment.train_and_eval()  # 进行训练和评估

    logger.info("best epoch: {}, best result:".format(experiment.best_epoch))
    acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l = experiment.best_result
    logger.info("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f} ".format(experiment.top_k, acc_l2r, mean_l2r, mrr_l2r))
    logger.info("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f} \n".format(experiment.top_k, acc_r2l, mean_r2l, mrr_r2l))

    logger.info("optimization finished!")  # 训练完成后将信息打印到日志中
    logger.info("total time elapsed: {:.4f} s".format(time.time() - t_total))  # 计算总共耗费的时间并将其打印到日志中
