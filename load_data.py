import time
import numpy as np

class AlignmentData:

    def __init__(self, data_dir="data/DBP15K/zh_en", train_rate=0.3, val_rate=0.0):
        t_ = time.time()

        self.train_rate = train_rate
        self.val_rate = val_rate
        # 读取 ent_ids_
        # {实体名称:实体ID} {实体ID:实体名称} [图1的所有实体ID, 图2的所有实体ID]
        self.ent2id_dict, self.id2ent_dict, [self.kg1_ent_ids, self.kg2_ent_ids] = self.load_dict(data_dir + "/ent_ids_", file_num=2)
        
        # 读取 rel_ids_
        # {关系名称:关系ID} {关系ID:关系名称} [图1的所有关系ID, 图2的所有关系ID]
        self.rel2id_dict, self.id2rel_dict, [self.kg1_rel_ids, self.kg2_rel_ids] = self.load_dict(data_dir + "/rel_ids_", file_num=2)

        # 总实体数
        self.ent_num = len(self.ent2id_dict)
        # 总关系数
        self.rel_num = len(self.rel2id_dict)

        # 读取 三元组triples_ 和 预对齐ill_ent_ids,都是元组
        self.triple_idx = self.load_triples(data_dir + "/triples_", file_num=2)
        self.ill_idx = self.load_triples(data_dir + "/ill_ent_ids", file_num=1)

        # 把预对齐按 train_rate, val_rate 分为 train+val+test
        np.random.shuffle(self.ill_idx)
        self.ill_train_idx, self.ill_val_idx, self.ill_test_idx = \
            np.array(self.ill_idx[:int(len(self.ill_idx) // 1 * train_rate)], dtype=np.int32), \
            np.array(self.ill_idx[int(len(self.ill_idx) // 1 * train_rate) : int(len(self.ill_idx) // 1 * (train_rate+val_rate))], dtype=np.int32), \
            np.array(self.ill_idx[int(len(self.ill_idx) // 1 * (train_rate+val_rate)):], dtype=np.int32)

        # 获取稀疏图,没有去除自环,没有加入反向边,没有交换ill
        # [[head, tail], ...] [[1], ...]             [[r], ...]
        self.sparse_edges_idx, self.sparse_values, self.sparse_rels_idx = self.gen_sparse_graph_from_triples()
        
        self.init_time = time.time() - t_

    def load_triples(self, data_dir, file_num=2):
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
        triple = []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [tuple(map(int, i.split("\t"))) for i in data]
                triple += data
        np.random.shuffle(triple)
        return triple

    def load_dict(self, data_dir, file_num=2):
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
        what2id, id2what, ids = {}, {}, []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [i.split("\t") for i in data]
                what2id = {**what2id, **dict([[i[1], int(i[0])] for i in data])}
                id2what = {**id2what, **dict([[int(i[0]), i[1]] for i in data])}
                ids.append(set([int(i[0]) for i in data]))
        return what2id, id2what, ids

    def gen_sparse_graph_from_triples(self):
        edge_dict = {}
        #print(self.triple_idx[0], self.triple_idx[1])
        for (h, r, t) in self.triple_idx:
            if (h, t) not in edge_dict:
                edge_dict[(h, t)] = []
            edge_dict[(h, t)].append(r)
        edges = [[h, t] for (h, t) in edge_dict for r in edge_dict[(h, t)]]
        values = [1 for (h, t) in edge_dict for r in edge_dict[(h, t)]]
        r_ij = [r for (h, t) in edge_dict for r in edge_dict[(h, t)]]
        edges = np.array(edges, dtype=np.int32)
        values = np.array(values, dtype=np.float32)
        r_ij = np.array(r_ij, dtype=np.float32)
        #print(len(edges), len(values), len(r_ij), len(self.triple_idx))
        return edges, values, r_ij

    def __repr__(self): # print 时调用
        return self.__class__.__name__ + " dataset summary:" + \
            "\n\tent_num: " + str(self.ent_num) + \
            "\n\trel_num: " + str(self.rel_num) + \
            "\n\ttriple_idx: " + str(len(self.triple_idx)) + \
            "\n\ttrain_rate: " + str(self.train_rate) + "\tval_rate: " + str(self.val_rate) + \
            "\n\till_idx(train/test/val): " + str(len(self.ill_idx)) + " = " + str(len(self.ill_train_idx)) + " + " + str(len(self.ill_test_idx)) + " + " + str(len(self.ill_val_idx)) + \
            "\n\tsprase_edges_idx: " + str(len(self.sparse_edges_idx)) + \
            "\n\t----------------------------- init_time: " + str(round(self.init_time, 3)) + "s"


if __name__ == '__main__':
    
    # TEST
    d = AlignmentData()
    print(d)
    #print(d.triple_idx[0], d.sparse_edges_idx[0], d.sparse_rels_idx[0]) (14688, 590, 16302) [14688 16302] 590.0