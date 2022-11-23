from datanetAPI import DatanetAPI
import pandas as pd
import networkx as nx

api = DatanetAPI(f'../data/scalability/test/')
it = iter(api)

delay = []
intensity = []
topo_size = []
file = []
num_sample = 0
for sample in it:
    aux_delay = []
    aux_topo_size = []
    aux_file = []
    G = nx.DiGraph(sample.get_topology_object())
    T = sample.get_traffic_matrix()
    P = sample.get_performance_matrix()
    R = sample.get_routing_matrix()
    f = sample._get_data_set_file_name()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                for f_id in range(len(T[src, dst]['Flows'])):
                    if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                        aux_delay.append(P[src, dst]['Flows'][f_id]['Jitter']/P[src, dst]['Flows'][f_id]['AvgDelay'])
                        aux_topo_size.append(G.number_of_nodes())
                        aux_file.append(f)
    if all(i > 0 for i in aux_delay):
        print(num_sample)
        num_sample += 1
        delay.extend(aux_delay)
        topo_size.extend(aux_topo_size)
        file.extend(aux_file)
    else:
        continue

df = pd.DataFrame(
    {"jitter": delay, "topo_size": topo_size, "file": file})
df.to_feather(f"scalability_test_jitter_dataframe")
