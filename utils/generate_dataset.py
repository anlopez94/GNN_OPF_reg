import pandapower as pp
import os
import networkx as nx
from networkx.readwrite import json_graph
import json
import tqdm
import tensorflow as tf
tf.config.run_functions_eagerly(run_eagerly=True)
import json
import pickle
import networkx as nx
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from random import randint, uniform
import joblib
import pandapower.networks as nw







class LoadDataSave(object):

    def __init__(self,
                 case_name='case9',
                 n=100,
                 nodes_change=0.15,
                 load_change_sup=0.1,
                 load_change_inf=0.4,
                 moving_edges=False,
                 base_dir='datasets',
                 use_file_tranformer=False):
        self.base_dir = base_dir
        self.case_name = case_name
        self.dataset_dirs = []
        self.n = n
        self.use_file_tranformer = use_file_tranformer
        self.nodes_change = nodes_change
        self.load_change_sup = load_change_sup
        self.load_change_inf = load_change_inf
        self.moving_edges = moving_edges

        self.file = self.base_dir + '/' + self.case_name + '/' + self.case_name + '.xlsx'
        self.create_directory()

        data = self.load_data()
        self.save_data_pkl(data)

    def create_directory(self):

        load_change_sup = 'load_change_sup' + str(self.load_change_sup)
        load_change_inf = 'load_change_inf' + str(self.load_change_inf)
        nodes_change = 'nodes_change' + str(self.nodes_change)
        moving_edges = 'moving_edges' + str(self.moving_edges)
        data_folder = ('-').join([load_change_sup, load_change_inf, nodes_change, moving_edges])
        self.path = self.base_dir + '/' + self.case_name + '/' + data_folder + '/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def load_data(self):
        # load excel
        self.net_base = pp.from_excel(self.file, convert=True)
        self.nodes = self.net_base['bus'].index.values.tolist()
        self._load_gen_config(self.net_base)
        data_to_file = []
        nodes_info_all = []
        gen_all = []
        data_all = []
        for i in range(self.n):
            net = False
            while net==False:
                #change load profile until solution convergence
                net = copy.deepcopy(self.net_base)
                net = self.create_load(net)
                net_aux = copy.deepcopy(net)
                net = self.run_optimalpowerflow(net)

            # net_aux = self.run_dcoptimalpowerflow(net_aux)
            # edges, nodes_info, weights, gen_dcopf = self.create_net_info(net_aux)
            # gen = self.create_gen_info(net)
            # nodes_info_all.append(nodes_info)
            # gen_all.append(gen)
            # data_all.append([edges, nodes_info, weights, gen, gen_dcopf])

            #el dc no nos da valores de voltaje ni de la q, asi que metemos directamete el optimo
            print(f'creating: {i}')
            net_aux = self.run_dcoptimalpowerflow(net_aux)
            edges, nodes_info, weights, gen = self.create_net_info(net)
            gen_dcopf = self.create_gen_info(net_aux)
            nodes_info_all.append(nodes_info)
            gen_all.append(gen)
            data_all.append([edges, nodes_info, weights, gen, gen_dcopf])
        if self.use_file_tranformer == True:
            self.load_transformer()
        else:
            self.preprocess_data(nodes_info_all, gen_all, weights)
            self.save_transformer()
        # convert to tensor and save in the file
        for i in range(0, len(data_all)):
            print(f'saving: {i}')
            edges, nodes_info, weights, gen, gen_dcopf = data_all[i]
            edges, node_features, weights, gen, gen_nn, gen_dcopf = self.convert_net_info(edges, nodes_info, weights, gen, gen_dcopf)
            data_to_file.append({'edges': edges, 'node_features': node_features, 'weights': weights, 'gen': gen, 'gen_nn': gen_nn, 'gen_dcopf': gen_dcopf})
        return data_to_file





    def preprocess_data(self, nodes, gen, weights):
        x = []
        for i in range(len(nodes)):
            instance_nodes = list(nodes[i].values())
            x.extend(instance_nodes)
        x = np.array(x)
        y = []
        for i in range(len(gen)):
            instance_gen = list(gen[i].values())
            y.append(instance_gen)
        y = np.array(y)
        weights = np.asarray(weights).astype('float32')
        self.scaler_nodes = MinMaxScaler()
        self.scaler_nodes.fit(x)
        x = self.scaler_nodes.transform(x)
        self.scaler_gen = MinMaxScaler()
        self.scaler_gen.fit(y)
        y = self.scaler_gen.transform(y)
        self.scaler_weights = MinMaxScaler()
        self.scaler_weights.fit(weights)
        weights = self.scaler_weights.transform(weights)
        pass

    def save_transformer(self):
        scaler_filename = self.path + '/scaler_nodes'
        joblib.dump(self.scaler_nodes, scaler_filename)

        scaler_filename = self.path + '/scaler_gen'
        joblib.dump(self.scaler_gen, scaler_filename)

    def load_transformer(self):
        scalar_x_filename = self.path + '/scaler_nodes'
        self.scaler_nodes = joblib.load(scalar_x_filename)
        scalar_y_filename = self.path + '/scaler_gen'
        self.scaler_gen = joblib.load(scalar_y_filename)

    def create_load(self, net):
        """
        :return:
        :rtype:
        """
        loads = net.load['p_mw'].to_list()
        loads_to_change = int(round(len(loads)*self.nodes_change,0))
        for i in range(max(loads_to_change, 1)):
            index = randint(0, len(loads) - 1)
            percentage = uniform(1-self.load_change_inf, 1+self.load_change_sup)
            loads[index] = loads[index] * percentage
        net.load.drop(columns='p_mw', inplace=True)
        net.load['p_mw'] = loads
        return net


    def create_net_info(self, net):
        node_columns = ['vm_pu', 'va_degree', 'p_mw', 'q_mvar']
        line = ['r_ohm_per_km', 'x_ohm_per_km']
        nodes_info, nodes_neighbour = {}, {}
        weights, edges = [], []

        # node x information
        for index, row in net['res_bus'].iterrows():
            nodes_info[index] = row.loc[node_columns].values
        # line information
        for index, row in net['line'].iterrows():
            if self.moving_edges:
                p_line = net.res_line.loc[index]['p_from_mw']
                if p_line > 0:
                    edges.append([row['from_bus'], row['to_bus']])
                else:
                    edges.append([row['to_bus'], row['from_bus']])
                weights.append(row.loc[line].values)
            else:
                edges.append([row['from_bus'], row['to_bus']])
                edges.append([row['to_bus'], row['from_bus']])
                weights.append(row.loc[line].values)
                weights.append(row.loc[line].values)

        gen_dcopf = self.create_gen_info(net)
        return [edges, nodes_info, weights, gen_dcopf]

    def create_gen_info(self, net):
        gen = {}
        for index, row in net['res_gen'].iterrows():
            gen[net['gen'].iloc[index]['bus']] = row['p_mw']
        return gen



    def convert_net_info(self, edges, nodes_info, weights, gen, gen_dcopf):
        node_features_pre = list(nodes_info.values())
        node_features_pre = self.scaler_nodes.transform(node_features_pre)
        node_features = tf.cast(
            np.asarray(node_features_pre), dtype=tf.dtypes.float32
        )
        edges = np.asarray(edges).T
        weights = np.asarray(weights).astype('float32')
        weights = self.scaler_weights.transform(weights)
        weights = tf.cast(
            weights, dtype=tf.dtypes.float32
        )
        gen_nn = copy.deepcopy(gen)
        gen_values = [list(gen.values())]
        gen_values_norm = self.scaler_gen.transform(gen_values)
        gen = dict(zip(gen.keys(), gen_values_norm[0].tolist()))

        return [edges, node_features, weights, gen, gen_nn, gen_dcopf]

    def _load_gen_config(self, net):
        """

        """
        self.nodex_gen = []
        for index, row in net['gen'].iterrows():
            self.nodex_gen.append(row['bus'])

    def run_powerflow(self, net):
        """
        Calcula el power flow en la red self.net
        :return:
        :rtype:
        """
        try:
            pp.runpp(net)
            return net
        except pp.powerflow.LoadflowNotConverged:
            return False
        return net


    def run_dcoptimalpowerflow(self, net):
        """
        Calcula el power flow en la red self.net
        :return:
        :rtype:
        """

        pp.rundcopp(net)
        return net

    def run_optimalpowerflow(self, net):
        """
        Calcula el power flow en la red self.net
        :return:
        :rtype:
        """
        try:
            pp.runopp(net)
            return net
        except pp.optimal_powerflow.OPFNotConverged:
            return False


    def save_data_pkl(self, data):
        name = self.path + '/data.pkl'
        with open(name, 'wb') as f:
            pickle.dump(data, f)



class LoadNetSaveExcel(object):

    def __init__(self,
                 net,
                 case_name='case9',
                 base_dir='datasets'):
        self.base_dir = base_dir
        self.case_name = case_name
        self.net = net

    def save_excel(self):

        pp.to_excel(self.net, self.base_dir + '/' + self.case_name + '/' + self.case_name+'.xlsx')
        solution = pp.runpp(self.net)
        pp.to_excel(self.net, self.base_dir + '/' + self.case_name + '/' + self.case_name + '_pf.xlsx')
        solution = pp.runopp(self.net)
        pp.to_excel(self.net, self.base_dir + '/' + self.case_name + '/' + self.case_name + '_opf.xlsx')