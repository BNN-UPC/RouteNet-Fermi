from datanetAPI import DatanetAPI
import pandas as pd
import networkx as nx

api = DatanetAPI(f'../data/scheduling/test')
it = iter(api)

delay = []
jitter = []
loss = []
intensity_delay = []
intensity_jitter = []
intensity_loss = []
num_sample = 0
num_sample_delay = 0
num_sample_jitter = 0
num_sample_loss = 0
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
    if all(i > 0 for i in aux_delay):
        delay.extend(aux_delay)
        intensity_delay.extend(aux_intensity)
        num_sample_delay += 1
    if all(i > 0 for i in aux_jitter):
        jitter.extend(aux_jitter)
        intensity_jitter.extend(aux_intensity)
        num_sample_jitter += 1
    if not all(i < 0.01 for i in aux_loss):
        loss.extend(aux_loss)
        intensity_loss.extend(aux_intensity)
        num_sample_loss += 1
    num_sample+=1
    print(num_sample)

print(num_sample_delay)
print(num_sample_jitter)
print(num_sample_loss)

df = pd.DataFrame(
    {"delay": delay, "intensity": intensity_delay})
df.to_feather(f"scheduling_test_dataframe_delay")

df = pd.DataFrame(
    {"jitter": jitter, "intensity": intensity_jitter})
df.to_feather(f"scheduling_test_dataframe_jitter")

df = pd.DataFrame(
    {"loss": loss, "intensity": intensity_loss})
df.to_feather(f"scheduling_test_dataframe_loss")
