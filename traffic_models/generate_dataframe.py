from datanetAPI import DatanetAPI
import pandas as pd
import networkx as nx

for tm in ['constant_bitrate', 'onoff', 'autocorrelated', 'modulated', 'all_multiplexed']:
    api = DatanetAPI(f'../data/traffic_models/{tm}/test/')
    it = iter(api)

    delay = []
    jitter = []
    loss = []
    intensity = []
    num_sample = 0
    for sample in it:
        aux_delay = []
        aux_jitter = []
        aux_loss = []
        aux_intensity = []
        G = nx.DiGraph(sample.get_topology_object())
        T = sample.get_traffic_matrix()
        P = sample.get_performance_matrix()
        R = sample.get_routing_matrix()
        f = sample._get_data_set_file_name()
        inten = sample.get_maxAvgLambda()
        for src in range(G.number_of_nodes()):
            for dst in range(G.number_of_nodes()):
                if src != dst:
                    for f_id in range(len(T[src, dst]['Flows'])):
                        if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                            aux_delay.append(P[src, dst]['Flows'][f_id]['AvgDelay'])
                            aux_jitter.append(P[src, dst]['Flows'][f_id]['Jitter'])
                            aux_loss.append(P[src, dst]['Flows'][0]['PktsDrop'] / T[src, dst]['AggInfo']['PktsGen'])
                            aux_intensity.append(inten)
        if all(i > 0 for i in aux_delay) and all(i >= 0 for i in aux_jitter) and all(i >= 0 for i in aux_loss):
            print(num_sample)
            num_sample += 1
            delay.extend(aux_delay)
            jitter.extend(aux_jitter)
            loss.extend(aux_loss)
            intensity.extend(aux_intensity)
        else:
            continue

    df = pd.DataFrame(
        {"delay": delay, "jitter": jitter, "loss": loss, "intensity": intensity})
    df.to_feather(f"{tm}_test_dataframe")

