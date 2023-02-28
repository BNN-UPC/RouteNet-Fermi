from datanetAPI import DatanetAPI
import pandas as pd
import networkx as nx

TOPOLOGY = nx.read_gml('./layer2/2TG/layer2_graph_2tg.txt').to_directed()

LAYER2_PATH = {}
with open('./layer2/2TG/layer2_routing_2tg.txt') as file:
    for line in file:
        data = line.rstrip().split()
        src = int(data[0])
        dst = int(data[1])
        path = [int(i.replace('[', '').replace(']', '').replace(',', '')) for i in data[2:]]
        l2_path = []
        for n in path:
            if TOPOLOGY.nodes[str(n)]['type'] == 's':
                l2_path.append(n)

        LAYER2_PATH[(src, dst)] = l2_path

api = DatanetAPI(f'../data/1CBR-1400B-2TG_1/test/')
it = iter(api)

delay = []
intensity = []
ipg = []
traffic = []
burst_size = []
path_len = []
off_time = []
packet_size = []
sample_id = []
routing = []
load_sequence = []
op_sequence = []
sample_file = []
input_file = []
num_sample = 0
for sample in it:
    ip_dict = {}
    op_dict = {}
    aux_delay = []
    aux_ipg = []
    aux_traffic = []
    aux_burst_size = []
    aux_off_time = []
    aux_packet_size = []
    aux_path_len = []
    aux_routing = []
    aux_intensity = []
    aux_op_sequence = []
    aux_load_sequence = []
    G = nx.DiGraph(sample.get_topology_object())
    T = sample.get_traffic_matrix()
    P = sample.get_performance_matrix()
    R = sample.get_routing_matrix()
    s_file = sample._get_data_set_file_name()
    i_file = sample._input_files_line

    PATH_TO_TG = {}
    with open(f'./layer2/2TG/path-to-2tg-{len(G)}.txt') as file:
        for line in file:
            src_router, dst_router, src_tg, dst_tg = [int(i) for i in line.rstrip().split()]
            PATH_TO_TG[(src_router, dst_router)] = (src_tg, dst_tg)

    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                if T[src, dst]['AggInfo']['AvgBw'] != 0 and T[src, dst]['AggInfo']['PktsGen'] != 0:
                    for f_id in range(len(T[src, dst]['Flows'])):
                        if T[src, dst]['AggInfo']['AvgBw'] != 0 and T[src, dst]['AggInfo']['PktsGen'] != 0:
                            aux_delay.append(P[src, dst]['AggInfo']['AvgDelay'] * 1000)
                            aux_ipg.append(T[src, dst]["AggInfo"]["meanIPG"])
                            aux_traffic.append(T[src, dst]['AggInfo']['AvgBw'])
                            aux_burst_size.append(T[src, dst]['AggInfo']['AvgBurstSize'])
                            aux_off_time.append(T[src, dst]['AggInfo']['AvgOffTime'])
                            aux_packet_size.append(T[src, dst]['AggInfo']['AvgPktSize'])
                            aux_path_len.append(len(R[src, dst]))
                            aux_routing.append(R[src, dst])

                            l3_routing = R[src, dst]
                            l2_routing = [d for d in LAYER2_PATH[(PATH_TO_TG[(src, dst)][0], src)]]
                            for r_1, r_2 in [l3_routing[i:i + 2] for i in range(0, len(l3_routing) - 1)]:
                                l2_routing.append(r_1)
                                l2_routing.extend(LAYER2_PATH[(r_1, r_2)])
                            l2_routing.append(r_2)
                            l2_routing.extend(LAYER2_PATH[(dst, PATH_TO_TG[(src, dst)][1])])
                            l2_routing.append(PATH_TO_TG[(src, dst)][1])
                            # # print("l2_routing: ", l2_routing)

                            l3_routing = [PATH_TO_TG[(src, dst)][0]]
                            l3_routing.extend(R[src, dst])
                            l3_routing.append(PATH_TO_TG[(src, dst)][1])

                            # # print("l3_routing: ", l3_routing)

                            router_count = 1
                            pos = 0
                            for d_1, d_2 in [l2_routing[i:i + 2] for i in range(0, len(l2_routing) - 1)]:
                                # # print("d_1: ", d_1)
                                # # print("d_2: ", d_2)
                                device_type_d_1 = TOPOLOGY.nodes[str(d_1)]['type']
                                device_type_d_2 = TOPOLOGY.nodes[str(d_2)]['type']
                                if device_type_d_2 == 'tg':
                                    d_2 = 'tg'
                                elif device_type_d_1 == 'tg':
                                    d_1 = 'tg'

                                if device_type_d_1 == 'r':
                                    router_count += 1
                                    dst_router = l3_routing[router_count]
                                    if TOPOLOGY.nodes[str(dst_router)]['type'] == 'tg':
                                        dst_router = 'tg'
                                    # # print("router_count: ", router_count)
                                    # # print("dst_router: ", dst_router)
                                    op = '{}_op_{}_{}_{}'.format(device_type_d_1, d_1, d_1, dst_router)
                                elif device_type_d_2 == 'r':
                                    src_router = l3_routing[router_count - 1]

                                    if TOPOLOGY.nodes[str(src_router)]['type'] == 'tg':
                                        src_router = 'tg'
                                    # # print("router_count: ", router_count)
                                    # # print("src_router: ", src_router)
                                    op = '{}_op_{}_{}_{}'.format(device_type_d_1, d_1, src_router, d_2)
                                else:
                                    op = '{}_op_{}_{}'.format(device_type_d_1, d_1, d_2)

                                if device_type_d_1 == 'r':
                                    bw = 1e9
                                elif device_type_d_1 == 's':
                                    if device_type_d_2 == 's':
                                        bw = 80e9
                                    elif device_type_d_2 == 'tg':
                                        bw = 10e9
                                    elif device_type_d_2 == 'r':
                                        bw = 1e9

                                if op not in op_dict:
                                    op_dict[op] = T[src, dst]['AggInfo']['AvgBw'] / bw
                                else:
                                    op_dict[op] += T[src, dst]['AggInfo']['AvgBw'] / bw

    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                ops = []
                loads = []
                if T[src, dst]['AggInfo']['AvgBw'] != 0 and T[src, dst]['AggInfo']['PktsGen'] != 0:
                    for f_id in range(len(T[src, dst]['Flows'])):
                        if T[src, dst]['AggInfo']['AvgBw'] != 0 and T[src, dst]['AggInfo']['PktsGen'] != 0:
                            l3_routing = R[src, dst]
                            l2_routing = [d for d in LAYER2_PATH[(PATH_TO_TG[(src, dst)][0], src)]]
                            for r_1, r_2 in [l3_routing[i:i + 2] for i in range(0, len(l3_routing) - 1)]:
                                l2_routing.append(r_1)
                                l2_routing.extend(LAYER2_PATH[(r_1, r_2)])
                            l2_routing.append(r_2)
                            l2_routing.extend(LAYER2_PATH[(dst, PATH_TO_TG[(src, dst)][1])])
                            l2_routing.append(PATH_TO_TG[(src, dst)][1])
                            # # print("l2_routing: ", l2_routing)

                            l3_routing = [PATH_TO_TG[(src, dst)][0]]
                            l3_routing.extend(R[src, dst])
                            l3_routing.append(PATH_TO_TG[(src, dst)][1])

                            # # print("l3_routing: ", l3_routing)

                            router_count = 1
                            pos = 0
                            for d_1, d_2 in [l2_routing[i:i + 2] for i in range(0, len(l2_routing) - 1)]:
                                # # print("d_1: ", d_1)
                                # # print("d_2: ", d_2)
                                device_type_d_1 = TOPOLOGY.nodes[str(d_1)]['type']
                                device_type_d_2 = TOPOLOGY.nodes[str(d_2)]['type']
                                if device_type_d_2 == 'tg':
                                    d_2 = 'tg'
                                elif device_type_d_1 == 'tg':
                                    d_1 = 'tg'

                                if device_type_d_1 == 'r':
                                    router_count += 1
                                    dst_router = l3_routing[router_count]
                                    if TOPOLOGY.nodes[str(dst_router)]['type'] == 'tg':
                                        dst_router = 'tg'
                                    # # print("router_count: ", router_count)
                                    # # print("dst_router: ", dst_router)
                                    op = '{}_op_{}_{}_{}'.format(device_type_d_1, d_1, d_1, dst_router)
                                elif device_type_d_2 == 'r':
                                    src_router = l3_routing[router_count - 1]

                                    if TOPOLOGY.nodes[str(src_router)]['type'] == 'tg':
                                        src_router = 'tg'
                                    # # print("router_count: ", router_count)
                                    # # print("src_router: ", src_router)
                                    op = '{}_op_{}_{}_{}'.format(device_type_d_1, d_1, src_router, d_2)
                                else:
                                    op = '{}_op_{}_{}'.format(device_type_d_1, d_1, d_2)
                                ops.append(op)
                                loads.append(op_dict[op])
                            aux_op_sequence.append(ops)
                            aux_load_sequence.append(loads)

    if all(i > 0 for i in aux_delay) and len(aux_delay) > 0:
        print(num_sample)
        delay.extend(aux_delay)
        ipg.extend(aux_ipg)
        traffic.extend(aux_traffic)
        burst_size.extend(aux_burst_size)
        off_time.extend(aux_off_time)
        packet_size.extend(aux_packet_size)
        path_len.extend(aux_path_len)
        routing.extend(aux_routing)
        load_sequence.extend(aux_load_sequence)
        op_sequence.extend(aux_op_sequence)
        sample_id.extend([num_sample] * len(aux_delay))
        sample_file.extend([s_file] * len(aux_delay))
        input_file.extend([i_file] * len(aux_delay))
        intensity.extend([sample.get_max_link_load()] * len(aux_delay))
        num_sample += 1
    else:
        continue

print("len(sample_id)")
print(len(sample_id))
print("len(sample_file)")
print(len(sample_file))
print("len(input_file)")
print(len(input_file))
print("len(intensity)")
print(len(intensity))
print("len(ipg)")
print(len(ipg))
print("len(traffic)")
print(len(traffic))
print("len(packet_size)")
print(len(packet_size))
print("len(burst_size)")
print(len(burst_size))
print("len(off_time)")
print(len(off_time))
print("len(path_len)")
print(len(path_len))
print("len(routing)")
print(len(routing))
print("len(load_sequence)")
print(len(load_sequence))
print("len(op_sequence)")
print(len(op_sequence))
print("len(delay)")
print(len(delay))

df = pd.DataFrame(
    {"sample_id": sample_id,
     "sample_file": sample_file,
     "input_file": input_file,
     "intensity": intensity,
     "ipg": ipg,
     "traffic": traffic,
     "packet_size": packet_size,
     "burst_size": burst_size,
     "off_time": off_time,
     "path_len": path_len,
     "routing": routing,
     "load_sequence": load_sequence,
     "op_sequence": op_sequence,
     "delay": delay})
print(df)
df.to_feather(f"test_dataframe_1CBR-1400B-2TG")
