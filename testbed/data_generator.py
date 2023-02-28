"""
   Copyright 2020 Universitat Politècnica de Catalunya
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import tensorflow as tf
import random
import math
import networkx as nx
from datanetAPI import DatanetAPI

EXTERNAL_DISTRIBUTIONS = ['AR1-0', 'AR1-1']
SIZE_DIST = ['delay_dns_0.pcap', 'dns_0.pcap', 'rtp_160_0.pcap', 'delay_rtp_250k_0_0.pcap',
             'delay_video_call_rtp_0.pcap', 'delay_rtp_160k_1_1_1.pcap', 'delay_10_video_call_rtp_0.pcap',
             'delay_10_sip_0.pcap', 'delay_10_rtp_160k_0.pcap', 'rtp_250k_1_0.pcap', 'delay_10_rtp_250k_0_0.pcap',
             'video_call_0.pcap', 'sip_0.pcap', 'rtp_160_1.pcap', 'delay_10_video_call_0.pcap',
             'video_rtp_1588_0.pcap', 'delay_sip_0.pcap', 'delay_rtp_160k_1_1_0.pcap', 'delay_video_call_0.pcap',
             'delay_rtp_250k_2_0.pcap', 'delay_10_dns_0.pcap', 'delay_10_rtp_160k_1.pcap', 'rtp_250k_2_0.pcap',
             'delay_10_rtp_250k_1_0.pcap']
TIME_DIST = ['c', 's']

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


def generator(data_dir, shuffle=False):
    try:
        data_dir = data_dir.decode('UTF-8')
    except (UnicodeDecodeError, AttributeError):
        pass
    tool = DatanetAPI(data_dir, shuffle=shuffle)
    it = iter(tool)
    for sample in it:
        """print("Sample File: {}".format(sample._get_data_set_file_name()))
        print("Input File Line: {}".format(sample._input_files_line))"""
        G_copy = sample.get_topology_object().copy()
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        D = sample.get_performance_matrix()
        HG = network_to_hypergraph(network_graph=G_copy,
                                   routing_matrix=R,
                                   traffic_matrix=T,
                                   performance_matrix=D,
                                   sample=sample)

        ret = hypergraph_to_input_data(HG)
        if not all(d[0] > 0 for d in ret[1]):
            print("Error: negative delay when there is traffic.")
            continue
        if not len(ret[1]) > 0:
            continue

        """## print(ret[1])
        ## print(ret[2])"""
        yield ret


def network_to_hypergraph(network_graph, routing_matrix, traffic_matrix, performance_matrix, sample):
    G = nx.DiGraph(network_graph)
    R = routing_matrix
    T = traffic_matrix
    D = performance_matrix
    D_G = nx.DiGraph()

    """for src in range(len(TOPOLOGY)):
        for dst in range(len(TOPOLOGY)):
            if src != dst and TOPOLOGY.has_edge(str(src), str(dst)):
                src_device_type = TOPOLOGY.nodes[str(src)]['type']
                dst_device_type = TOPOLOGY.nodes[str(dst)]['type']
                if src_device_type == 's' and dst_device_type == 's':
                    D_G.add_node('{}_op_{}_{}'.format(src_device_type, src, dst),
                                 s_capacity=TOPOLOGY[str(src)][str(dst)][0]['bandwidth'])
                else:
                    if src_device_type == 's':
                        for i in range(len(TOPOLOGY)):
                            D_G.add_node('{}_op_{}_{}_{}'.format(src_device_type, src, dst, i),
                                         s_capacity=TOPOLOGY[str(src)][str(dst)][0]['bandwidth'])
                    elif src_device_type == 'r':
                        for i in range(len(TOPOLOGY)):
                            # print(str(src), str(dst))
                            # print(TOPOLOGY[str(src)][str(dst)])
                            D_G.add_node('{}_op_{}_{}_{}'.format(src_device_type, src, src, i),
                                         r_capacity=TOPOLOGY[str(src)][str(dst)][0]['bandwidth'])"""

    for src, dst, d in TOPOLOGY.edges(data=True):
        src_device_type = TOPOLOGY.nodes[src]['type']
        dst_device_type = TOPOLOGY.nodes[dst]['type']
        if src_device_type == 's' or dst_device_type == 'tg':
            if dst_device_type == 's' or dst_device_type == 'tg':
                if dst_device_type == 'tg':
                    dst = 'tg'
                D_G.add_node('{}_op_{}_{}'.format(src_device_type, src, dst), s_capacity=d['bandwidth'])
            elif dst_device_type == 'r' or dst_device_type == 'tg':
                for i in range(len(TOPOLOGY)):
                    for j in range(len(TOPOLOGY)):
                        if i != 'tg':
                            if TOPOLOGY.nodes[str(i)]['type'] == 'tg':
                                i = 'tg'
                        D_G.add_node('{}_op_{}_{}_{}'.format(src_device_type, src, i, j), s_capacity=d['bandwidth'])
                        ## print('{}_op_{}_{}_{}'.format(src_device_type, src, i, j), d['bandwidth'])
        elif src_device_type == 'r':
            for i in range(len(TOPOLOGY)):
                if TOPOLOGY.nodes[str(i)]['type'] == 'tg':
                    i = 'tg'
                D_G.add_node('{}_op_{}_{}_{}'.format(src_device_type, src, src, i), r_capacity=d['bandwidth'])

    PATH_TO_TG = {}
    with open(f'./layer2/2TG/path-to-2tg-{len(G)}.txt') as file:
        for line in file:
            src_router, dst_router, src_tg, dst_tg = [int(i) for i in line.rstrip().split()]
            PATH_TO_TG[(src_router, dst_router)] = (src_tg, dst_tg)

    # print"LAYER2_PATH_1")
    # printLAYER2_PATH)
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            # print"LAYER2_PATH[(14, 0)]")
            # printLAYER2_PATH[(14, 0)])
            if src != dst:
                if T[src, dst]['AggInfo']['AvgBw'] != 0 and T[src, dst]['AggInfo']['PktsGen'] != 0:
                    f = T[src, dst]['Flows'][0]
                    ipg_list = T[src, dst]["AggInfo"]["pIPG_lst"]
                    ipg_list.append(T[src, dst]["AggInfo"]["varIPG"])
                    ipg_list.append(T[src, dst]["AggInfo"]["meanIPG"])
                    # ## print"Added p_{}_{}".format(src, dst))
                    D_G.add_node('p_{}_{}'.format(src, dst),
                                 traffic=T[src, dst]['AggInfo']['AvgBw'],
                                 packets=T[src, dst]['AggInfo']['PktsGen'],
                                 packet_size=[T[src, dst]['AggInfo']['PktsGen'],
                                              T[src, dst]['AggInfo']['AvgPktSize'],
                                              T[src, dst]['AggInfo']['p10PktSize'],
                                              T[src, dst]['AggInfo']['p20PktSize'],
                                              T[src, dst]['AggInfo']['p50PktSize'],
                                              T[src, dst]['AggInfo']['p80PktSize'],
                                              T[src, dst]['AggInfo']['p90PktSize'],
                                              T[src, dst]['AggInfo']['VarPktSize']],
                                 burst_size=[T[src, dst]['AggInfo']['AvgBurstSize'],
                                             T[src, dst]['AggInfo']['p5BurstSize'],
                                             T[src, dst]['AggInfo']['p95BurstSize'],
                                             T[src, dst]['AggInfo']['varBurstSize']],
                                 off_time=[T[src, dst]['AggInfo']['AvgOffTime'],
                                           T[src, dst]['AggInfo']['p5OffTime'],
                                           T[src, dst]['AggInfo']['p95OffTime'],
                                           T[src, dst]['AggInfo']['varOffTime']],
                                 ipg=ipg_list,
                                 size_distribution=int(f['SizeDist'].value),
                                 time_distribution=int(f['TimeDist'].value),
                                 delay=D[src, dst]['AggInfo']['AvgDelay'] * 1000)
                    """l3_routing = [PATH_TO_TG[(src, dst)][0]]
                    l3_routing.extend(R[src, dst])
                    l3_routing.append(PATH_TO_TG[(src, dst)][1])
                    ## print(l3_routing)"""
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
                        """edge = '{}_ip_{}'.format(device_type_d_1, d_1), 'p_{}_{}'.format(src, dst)
                        if not D_G.has_edge(*edge):
                            D_G.add_edge(*edge, position=[pos])
                            # # print("Added edge: ", edge, " with position: ", pos)
                        else:
                            D_G[edge[0]][edge[1]]['position'].append(pos)
                            # # print("Added edge: ", edge, " with position: ", pos)
                        D_G.add_edge(*edge[::-1])"""

                        if device_type_d_1 == 'r':
                            router_count += 1
                            dst_router = l3_routing[router_count]
                            if TOPOLOGY.nodes[str(dst_router)]['type'] == 'tg':
                                dst_router = 'tg'
                            # # print("router_count: ", router_count)
                            # # print("dst_router: ", dst_router)
                            edge = (
                                '{}_op_{}_{}_{}'.format(device_type_d_1, d_1, d_1, dst_router),
                                'p_{}_{}'.format(src, dst))
                        elif device_type_d_2 == 'r':
                            src_router = l3_routing[router_count - 1]

                            if TOPOLOGY.nodes[str(src_router)]['type'] == 'tg':
                                src_router = 'tg'
                            # # print("router_count: ", router_count)
                            # # print("src_router: ", src_router)
                            edge = (
                                '{}_op_{}_{}_{}'.format(device_type_d_1, d_1, src_router, d_2),
                                'p_{}_{}'.format(src, dst))
                        else:
                            edge = ('{}_op_{}_{}'.format(device_type_d_1, d_1, d_2), 'p_{}_{}'.format(src, dst))

                        if not D_G.has_edge(*edge):
                            D_G.add_edge(*edge, position=[pos])
                            # # print("Added edge: ", edge, " with position: ", pos+1)
                        else:
                            D_G[edge[0]][edge[1]]['position'].append(pos)
                            # # print("Added edge: ", edge, " with position: ", pos+1)
                        D_G.add_edge(*edge[::-1])
                        pos += 1
                    # ## print('-' * 10)

    """# print("Removed nodes")
    # print([node for node, out_degree in D_G.out_degree() if out_degree == 0])"""
    D_G.remove_nodes_from([node for node, out_degree in D_G.out_degree() if out_degree == 0])

    """for n, data in D_G.nodes(data=True):
        if n.startswith('s_op'):
            print(n, data)"""
    return D_G


def hypergraph_to_input_data(hypergraph):
    n_p = 0
    n_s_op = 0
    n_r_op = 0
    mapping = {}
    for entity in list(hypergraph.nodes()):
        if entity.startswith('p'):
            mapping[entity] = ('p_{}'.format(n_p))
            n_p += 1
        elif entity.startswith('s_op'):
            # printentity)
            # printhypergraph.nodes(data=True)[entity])
            mapping[entity] = ('s_op_{}'.format(n_s_op))
            n_s_op += 1
        elif entity.startswith('r_op'):
            mapping[entity] = ('r_op_{}'.format(n_r_op))
            n_r_op += 1
    reversed_mapping = {v: k for k, v in mapping.items()}

    G = nx.relabel_nodes(hypergraph, mapping)

    r_op_to_path = []
    path_to_r_op = []

    s_op_to_path = []
    path_to_s_op = []
    ops_to_path = []
    for node in G.nodes:
        in_nodes = [s for s, d in G.in_edges(node)]
        if node.startswith('r_op_'):
            touple = []
            for n in in_nodes:
                path_pos = [d for s, d in G.out_edges(n)]
                touple.append([int(n.replace('p_', '')), path_pos.index(node)])
            path_to_r_op.append(touple)
        elif node.startswith('s_op_'):
            touple = []
            for n in in_nodes:
                path_pos = [d for s, d in G.out_edges(n)]
                touple.append([int(n.replace('p_', '')), path_pos.index(node)])
            path_to_s_op.append(touple)
        elif node.startswith('p_'):
            # ## print(node)
            rips = []
            rops = []
            sips = []
            sops = []
            path_len = 0
            # aux = []
            for n in in_nodes:
                positions = G[n][node]['position']
                path_len += len(positions)
                # aux.extend(positions)
            # ## print("aux")
            # ## print(aux)
            ips_ops = [-1] * path_len
            aux = [-1] * path_len
            # ## print("path_len")
            # ## print(path_len)
            for n in in_nodes:
                positions = G[n][node]['position']
                if n.startswith('s_op_'):
                    sops.append(int(n.replace('s_op_', '')))

                    for p in positions:
                        ips_ops[p] = int(n.replace('s_op_', ''))
                        aux[p] = n
                    # aux.append(n)
                elif n.startswith('r_op_'):
                    rops.append(int(n.replace('r_op_', '')))

                    for p in positions:
                        ips_ops[p] = int(n.replace('r_op_', '')) + n_s_op
                        aux[p] = n
                    # ips_ops.append(int(n.replace('r_op_', '')) + n_s_ip + n_s_op + n_r_ip)
            # print([reversed_mapping[d] for d in aux])
            # print([d for d in aux])
            ops_to_path.append(ips_ops)
            """all.append(sips)
            all.append([d+n_s_ip for d in sops])
            all.append([d+n_s_ip+n_s_op for d in rips])
            all.append([d+n_s_ip+n_s_op+n_r_ip for d in rops])"""
            s_op_to_path.append(sops)
            r_op_to_path.append(rops)

    ret = ({"traffic": np.expand_dims(list(nx.get_node_attributes(G, 'traffic').values()), axis=1),
            "packets": np.expand_dims(list(nx.get_node_attributes(G, 'packets').values()), axis=1),
            "packet_size": list(nx.get_node_attributes(G, 'packet_size').values()),
            "burst_size": list(nx.get_node_attributes(G, 'burst_size').values()),
            "off_time": list(nx.get_node_attributes(G, 'off_time').values()),
            "ipg": list(nx.get_node_attributes(G, 'ipg').values()),
            "size_distribution": list(nx.get_node_attributes(G, 'size_distribution').values()),
            "time_distribution": list(nx.get_node_attributes(G, 'time_distribution').values()),
            "s_capacity": np.expand_dims(list(nx.get_node_attributes(G, 's_capacity').values()), axis=1),
            "r_capacity": np.expand_dims(list(nx.get_node_attributes(G, 'r_capacity').values()), axis=1),
            "delay": np.expand_dims(list(nx.get_node_attributes(G, 'delay').values()), axis=1),
            "ops_to_path": tf.ragged.constant(ops_to_path),
            "r_op_to_path": tf.ragged.constant(r_op_to_path),
            "path_to_r_op": tf.ragged.constant(path_to_r_op, ragged_rank=1),
            "s_op_to_path": tf.ragged.constant(s_op_to_path),
            "path_to_s_op": tf.ragged.constant(path_to_s_op, ragged_rank=1)
            }, [[x] for x in list(nx.get_node_attributes(G, 'delay').values())], [1 if x > 1 else 1 for x in
                                                                                  list(nx.get_node_attributes(G,
                                                                                                              'delay').values())])
    # print("s_capacity")
    # print(ret[0]['s_capacity'])
    # print("r_capacity")
    # print(ret[0]['path_to_r_op'])
    # print('-'*200)
    return ret


def input_fn(data_dir, shuffle=False, samples=None):
    ds = tf.data.Dataset.from_generator(generator,
                                        args=[data_dir, shuffle],
                                        output_signature=(
                                            {"traffic": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "packets": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "packet_size": tf.TensorSpec(shape=(None, 8), dtype=tf.float32),
                                             "burst_size": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                             "off_time": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                             "ipg": tf.TensorSpec(shape=(None, 103), dtype=tf.float32),
                                             "size_distribution": tf.TensorSpec(shape=None, dtype=tf.int32),
                                             "time_distribution": tf.TensorSpec(shape=None, dtype=tf.int32),
                                             "s_capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "r_capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "delay": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "ops_to_path": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
                                             "r_op_to_path": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
                                             "path_to_r_op": tf.RaggedTensorSpec(shape=(None, None, 2), dtype=tf.int32,
                                                                                 ragged_rank=1),
                                             "s_op_to_path": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
                                             "path_to_s_op": tf.RaggedTensorSpec(shape=(None, None, 2), dtype=tf.int32,
                                                                                 ragged_rank=1)
                                             }
                                            , tf.TensorSpec(shape=(None), dtype=tf.float32)
                                            , tf.TensorSpec(shape=(None), dtype=tf.float32)
                                        ))

    if samples:
        ds = ds.take(samples)

    if shuffle:
        ds = ds.shuffle(500, reshuffle_each_iteration=True)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

