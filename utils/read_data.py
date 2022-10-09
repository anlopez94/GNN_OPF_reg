
import json
import pickle
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(run_eagerly=True)

"""
codigo hecho para importar redes de potencia, y convertirlo a los formatos necesarios para que las lean las gnn
tambien pruebas de como reconvertir con la accion para pasar a red que se pueda hacer power flow
funciones para generar la misma red con diferentes demandas para hacer un dataset de entrenamiento
"""




class LoadDataFromPkl(object):
    def __init__(self, case_name='case9', load_change_sup='0.1', load_change_inf='0.4', nodes_change=0.8, moving_edges=False, base_dir='datasets'):
        self.case_name = case_name
        self.base_dir = base_dir
        self.load_change_sup = load_change_sup
        self.load_change_inf = load_change_inf
        self.nodes_change = nodes_change
        self.moving_edges = moving_edges
        load_change_sup = 'load_change_sup' + str(self.load_change_sup)
        load_change_inf = 'load_change_inf' + str(self.load_change_inf)
        nodes_change = 'nodes_change' + str(self.nodes_change)
        moving_edges = 'moving_edges' + str(self.moving_edges)
        self.values_folder = ('-').join([load_change_sup, load_change_inf, nodes_change, moving_edges])
        self.pkl_path = self.base_dir + '/' + self.case_name + '/' + self.values_folder + '/data.pkl'
        self.x = []
        self.y = []
        self.data = None
        self.load_data()
        self.prepare_data()

    def load_data(self):

        if self.case_name =='case118':
            pkl_path = self.base_dir + '/' + self.case_name + '/' + self.values_folder + '/data1.pkl'
            with open(pkl_path, 'rb') as f:
                data1 = pickle.load(f)
            pkl_path = self.base_dir + '/' + self.case_name + '/' + self.values_folder + '/data2.pkl'
            with open(pkl_path, 'rb') as f:
                data2 = pickle.load(f)
            self.data = data1 + data2
        else:
            with open(self.pkl_path, 'rb') as f:
                self.data = pickle.load(f)
        return self.data

    def prepare_data(self):
        for data in self.data:
            if self.moving_edges:
                x = tf.reshape(data['node_features'], [data['node_features'].shape[0] * data['node_features'].shape[1]])
                # x = tf.convert_to_tensor([data['node_features'], data['edges']])

                edges = tf.convert_to_tensor(np.concatenate([data['edges'][0], data['edges'][1]]))
                edges = tf.cast(edges, tf.float32)
                x = tf.concat([x, edges], -1)
                pass
            else:
                x = data['node_features']
            self.x.append(x)
            self.y.append(list(data['gen'].values()))

    def divide_data(self, val=0.15, test=0.15):

        self.x_train = self.x[:int(len(self.x) * (1 - val - test))]
        self.y_train = self.y[:int(len(self.y) * (1 - val - test))]

        # self.x_train = self.x[int(len(self.x) * (1 - test)):]
        # self.y_train = self.y[int(len(self.y) * (1 - test)):]

        self.x_test = self.x[int(len(self.x) * (1 - test)):]
        self.y_test = self.y[int(len(self.y) * (1 - test)):]

        self.x_val = self.x[int(len(self.x) * (1 - val - test)):int(len(self.x) * (1 - test))]
        self.y_val = self.y[int(len(self.y) * (1 - val - test)):int(len(self.y) * (1 - test))]




def divide_data(case_name, load_change_sup='0.1', load_change_inf='0.1'):


    data = LoadDataFromPkl(case_name, load_change_sup, load_change_inf)

    data1 = data.data[:int(len(data.data) / 2)]
    data2 = data.data[int(len(data.data) / 2):]
    load_change_sup = 'load_change_sup' + str(load_change_sup)
    load_change_inf = 'load_change_inf' + str(load_change_inf)
    values_folder = ('-').join([load_change_sup, load_change_inf])

    name = 'datasets' + '/' + case_name + '/' + values_folder + '/data1.pkl'
    with open(name, 'wb') as f:
        pickle.dump(data1, f)

    name = 'datasets' + '/' + case_name + '/' + values_folder + '/data2.pkl'
    with open(name, 'wb') as f:
        pickle.dump(data2, f)




