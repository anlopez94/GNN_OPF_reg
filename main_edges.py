
from tensorflow import keras
from utils.gnn_new import GNN
from utils.read_data import LoadDataFromPkl
from utils.trainning_model import train_model, set_writer_and_checkpoint_dir, load_model, evaluate_model, save_results
import pickle


if __name__ == "__main__":


    case_name = 'case30'
    load_change_sup = '0.1'
    load_change_inf = '0.1'
    nodes_change = '0.8'
    moving_edges_list = [False]
    moving_edges = True
    message_iterations_list = [2]
    create_message_nn_list = [True]
    activation_fn_list = ['selu', 'tanh']
    activation_fn = 'selu'
    dropout_rate = 0
    dropout_rate_readout = 0.2
    dropout_rate_list = [0]
    for dropout_rate in dropout_rate_list:
        for message_iterations in message_iterations_list:
            for create_message_nn in create_message_nn_list:
                load_change_sup_path = 'load_change_sup' + str(load_change_sup)
                load_change_inf_path = 'load_change_inf' + str(load_change_inf)
                nodes_change_path = 'nodes_change' + str(nodes_change)
                moving_edges_path = 'moving_edges' + str(moving_edges)
                values_folder = ('-').join([load_change_sup_path, load_change_inf_path, nodes_change_path, moving_edges_path])
                path = case_name + '/' + values_folder
                Data = LoadDataFromPkl(case_name,  load_change_sup=load_change_sup, load_change_inf=load_change_inf, nodes_change=nodes_change, moving_edges=moving_edges)

                Data.divide_data(val=0.15, test=0.15)
                info = Data.data[0]
                graph_info = (info['node_features'], info['edges'], info['weights'], list(info['gen'].keys()))
                hidden_units_update = "128-16"
                hidden_units_readout = "32-16"
                hidden_units_message = '64-16'
                node_state_size = 16

                epochs = 450
                #necisitamos que sea uno por como esta montada la GNN
                batch_size = 1
                #batch size real
                batch_size2 = 500
                learning_rate = 0.003
                beta_1 = 0.9
                epsilon = 0.01
                optimizer = keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    beta_1=beta_1,
                    epsilon=epsilon)

                gnn_model = GNN(
                    graph_info=graph_info,
                    hidden_units_update=hidden_units_update,
                    hidden_units_readout=hidden_units_readout,
                    hidden_units_message=hidden_units_message,
                    activation_fn=activation_fn,
                    node_state_size=node_state_size,
                    final_activation_fn='linear',
                    dropout_rate=dropout_rate,
                    dropout_rate_readout=dropout_rate_readout,
                    message_iterations=message_iterations,
                    create_message_nn=create_message_nn,
                    name="gnn_model",
                    moving_edges=moving_edges,
                )
                gnn_model.build()
                gnn_model.summary()
                # gnn_model.build(graph_info)
                print("GNN output shape:", gnn_model(Data.x[0]))

                gnn_model.summary()
                writer = set_writer_and_checkpoint_dir(base_dir='logs', case_name=case_name, batch=batch_size, epochs=epochs, optimizer='ADAM')

                epochs_path = 'epochs' + str(epochs)
                batch_size2_path = 'batch_size2' + str(batch_size2)
                activation_fn_path = 'activation_fn' + str(activation_fn)
                create_message_nn_path = 'create_m' + str(create_message_nn)
                message_iterations_path = 'message_iter' + str(message_iterations)
                dropout_rate_path = 'dropout_rate' + str(dropout_rate)
                learning_rate_path = 'lr' + str(learning_rate)
                epsilon_path = 'ep' + str(epsilon)
                check_folder = ('-').join(
                    [epochs_path, batch_size2_path, activation_fn_path, create_message_nn_path, message_iterations_path,
                     learning_rate_path, epsilon_path, dropout_rate_path])
                checkpoint_path = path + '/' + check_folder

                losses, model = train_model(gnn_model, Data, 'mse', optimizer, epochs, batch_size, batch_size2, writer, True, checkpoint_path)

                #Validate model on test dataset
                model = load_model(checkpoint_path)
                test_loss, y_test, logits = evaluate_model(gnn_model, Data)
                print(losses)
                print(test_loss)
                results_y = {'y_test': y_test, 'logits': logits, 'test_loss': test_loss}
                save_results(checkpoint_path, 'results_y', results_y)
