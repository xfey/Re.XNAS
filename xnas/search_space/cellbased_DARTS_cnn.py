from xnas.search_space.cellbased_ops import *


class DartsCell(nn.Module):
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, basic_op_list):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        self.basic_op_list = basic_op_list

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, C, stride, self.basic_op_list)
                self.dag[i].append(op)

    def forward(self, s0, s1, sample):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        w_dag = darts_weight_unpack(sample, self.n_nodes)
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w)
                        for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)
        s_out = torch.cat(states[2:], 1)
        return s_out


class DartsCNN(nn.Module):

    def __init__(self, C=16, n_classes=10, n_layers=8, n_nodes=4, basic_op_list=[]):
        super().__init__()
        stem_multiplier = 3
        self.C_in = 3  # 3
        self.C = C  # 16
        self.n_classes = n_classes  # 10
        self.n_layers = n_layers  # 8
        self.n_nodes = n_nodes  # 4
        self.basic_op_list = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3',
                              'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none'] if len(basic_op_list) == 0 else basic_op_list
        self.non_op_idx = get_op_index(self.basic_op_list, NON_PARAMETER_OP)
        self.para_op_idx = get_op_index(self.basic_op_list, PARAMETER_OP)
        C_cur = stem_multiplier * C  # 3 * 16 = 48
        self.stem = nn.Sequential(
            nn.Conv2d(self.C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C
        # 48   48   16
        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False
            cell = DartsCell(n_nodes, C_pp, C_p, C_cur,
                             reduction_p, reduction, self.basic_op_list)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)
        # number of edges per cell
        self.num_edges = sum(list(range(2, self.n_nodes + 2)))
        self.num_ops = len(self.basic_op_list)
        # whole edges
        self.all_edges = 2 * self.num_edges
        self.norm_node_index = self._node_index(n_nodes, input_nodes=2, start_index=0)
        self.reduce_node_index = self._node_index(n_nodes, input_nodes=2, start_index=self.num_edges)

    def forward(self, x, sample):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = sample[self.num_edges:] if cell.reduction else sample[0:self.num_edges]
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits

    def genotype(self, theta):
        Genotype = namedtuple(
            'Genotype', 'normal normal_concat reduce reduce_concat')
        theta_norm = darts_weight_unpack(
            theta[0:self.num_edges], self.n_nodes)
        theta_reduce = darts_weight_unpack(
            theta[self.num_edges:], self.n_nodes)
        gene_normal = parse_from_numpy(
            theta_norm, k=2, basic_op_list=self.basic_op_list)
        gene_reduce = parse_from_numpy(
            theta_reduce, k=2, basic_op_list=self.basic_op_list)
        concat = range(2, 2+self.n_nodes)  # concat all intermediate nodes
        return Genotype(normal=gene_normal, normal_concat=concat,
                        reduce=gene_reduce, reduce_concat=concat)

    def genotype_to_onehot_sample(self, genotype):
        sample = np.zeros([self.all_edges, len(self.basic_op_list)])
        norm_gene = genotype[0]
        reduce_gene = genotype[2]
        num_select = list(range(2, 2+self.n_nodes))
        for j, _gene in enumerate([norm_gene, reduce_gene]):
            for i, node in enumerate(_gene):
                for op in node:
                    op_name = op[0]
                    op_id = op[1]
                    if i == 0:
                        true_id = op_id + j * self.num_edges
                    else:
                        if i == 1:
                            _temp = num_select[0]
                        else:
                            _temp = sum(num_select[0:i])
                        true_id = op_id + _temp + j * self.num_edges
                    sample[true_id, self.basic_op_list.index(op_name)] = 1
        for i in range(self.all_edges):
            if np.sum(sample[i, :]) == 0:
                sample[i, 7] = 1
        return sample

    def _node_index(self, n_nodes, input_nodes=2, start_index=0):
        node_index = []
        start_index = start_index
        end_index = input_nodes + start_index
        for i in range(n_nodes):
            node_index.append(list(range(start_index, end_index)))
            start_index = end_index
            end_index += input_nodes + i + 1
        return node_index


# build API

def _DartsCNN():
    from xnas.core.config import cfg
    return DartsCNN(
        C=cfg.SPACE.CHANNEL,
        n_classes=cfg.SPACE.NUM_CLASSES,
        n_layers=cfg.SPACE.LAYERS,
        n_nodes=cfg.SPACE.NODES,
        basic_op_list=cfg.SPACE.BASIC_OP)