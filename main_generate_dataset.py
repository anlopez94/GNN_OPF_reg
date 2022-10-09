from utils.generate_dataset import LoadDataSave
from utils.read_data import divide_data

if __name__ == "__main__":

    # loader.save_excel()
    case_name = 'case30'
    load_change_sup = 0.1
    load_change_inf = 0.1
    nodes_change = 0.8
    moving_edges = False
    moving_edges_list = [True, False]
    n = 10000
    for moving_edges in moving_edges_list:
        Data = LoadDataSave(case_name=case_name, base_dir='datasets', n=n, nodes_change=nodes_change, load_change_sup=load_change_sup,
                             load_change_inf=load_change_inf, moving_edges=moving_edges)
        pass
    # loader = LoadNetSaveExcel(case_name ='case30')
    # loader.save_excel()

    # divide_data(case_name='case9', load_change_inf=0.4, load_change_sup=0.1)