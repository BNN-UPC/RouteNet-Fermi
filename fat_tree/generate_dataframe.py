from datanetAPI import DatanetAPI
import pandas as pd
import networkx as nx

for N in [16, 64, 128]:
    print(f"Generating dataframe for N = {N}...")
    api = DatanetAPI(f'/data/TON23/fat{N}/test')
    it = iter(api)

    delay = []
    intensity = []
    topo_size = []
    source = []
    destintaion = []
    file = []
    sample_id = []
    num_sample = 0
    for sample in it:
        aux_delay = []
        aux_source = []
        aux_destintaion = []
        aux_sample_id = []
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
                            aux_delay.append(P[src, dst]['Flows'][f_id]['AvgDelay'])
                            aux_source.append(src)
                            aux_destintaion.append(dst)
                            aux_sample_id.append(num_sample)

        if all(i > 0 for i in aux_delay):
            print(num_sample)
            num_sample += 1
            delay.extend(aux_delay)
            source.extend(aux_source)
            destintaion.extend(aux_destintaion)
            sample_id.extend(aux_sample_id)
        else:
            continue

    df = pd.DataFrame(
        {"sample_id": sample_id, "source": source, "destination":destintaion, "delay": delay})
    df.to_feather(f"fattree{N}_test_dataframe")
