'''
 *
 * Copyright (C) 2020 Universitat Politècnica de Catalunya.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
'''

# -*- coding: utf-8 -*-

import os, tarfile, numpy, math, networkx, queue, random,traceback
from enum import Enum
import timeit


class SizeDist(Enum):
    """
    Enumeration of the supported size distributions
    """
    DETERMINISTIC_S = 0

class TimeDist(Enum):
    """
    Enumeration of the supported time distributions
    """
    CBR_T = 0
    MULTIBURST_T = 1


class DatanetException(Exception):
    """
    Exceptions generated when processing dataset
    """
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg


class Sample:
    """
    Class used to contain the results of a single iteration in the dataset
    reading process.

    ...

    Attributes
    ----------
    global_packets : double
        Overall number of packets transmitteds in network
    global_losses : double
        Overall number of packets lost in network
    global_delay : double
        Overall delay in network
    max_link_load: double
        This variable is used in our testbed to indicate the maximum utilization
        of the links of the testbed.
    performance_matrix : NxN matrix
        Matrix where each cell [i,j] contains aggregated
        information about transmission parameters between source i and
        destination j.
    traffic_matrix : NxN matrix
        Matrix where each cell [i,j] contains aggregated
        information about size and time distributions between source i and
        destination j.
    routing_matrix : NxN matrix
        Matrix where each cell [i,j] contains the path, if it exists, between
        source i and destination j.
    topology_object :
        Network topology using networkx format.
    """

    global_packets = None
    global_losses = None
    global_delay = None

    performance_matrix = None
    traffic_matrix = None
    routing_matrix = None
    topology_object = None

    data_set_file = None
    _results_line = None
    _input_files_line = None
    _traffic_ipg_line = None
    _routing_file = None
    _graph_file = None

    def get_global_packets(self):
        """
        Return the number of packets transmitted in the network per time unit of this Sample instance.
        """

        return self.global_packets

    def get_global_packets_for_tos(self,tos):
        """
        Return the number of packets with tos transmitted in the network per time unit of this Sample instance.
        """

        return self.global_qos_info[tos]["global_packets"]

    def get_global_losses(self):
        """
        Return the number of packets dropped in the network per time unit of this Sample instance.
        """

        return self.global_losses

    def get_global_losses_for_tos(self,tos):
        """
        Return the number of packets with tos dropped in the network per time unit of this Sample instance.
        """

        return self.global_qos_info[tos]["global_losses"]

    def get_global_delay(self):
        """
        Return the average per-packet delay over all the packets transmitted in the network in time units
        of this sample instance.
        """

        return self.global_delay


    def get_global_delay_for_tos(self,tos):
        """
        Return the average per-packet delay over all the packets with tos transmitted in the network in time units
        of this sample instance.
        """

        return self.global_qos_info[tos]["global_delay"]

    def get_capture_time(self):
        """
        Returns the duration of the capture process ignoring the transitory period where no packets are
        captured
        """

        return self.capture_time

    def get_max_link_load(self):
        """
        Returns the maximum utilization that can be reached of a link. It is used to determine the load of
        the network
        """

        return self.max_link_load


    def get_performance_matrix(self):
        """
        Returns the performance_matrix of this Sample instance.
        """

        return self.performance_matrix

    def get_srcdst_performance(self, src, dst):
        """


        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary
            Information stored in the Result matrix for the requested src-dst.

        """
        return self.performance_matrix[src, dst]

    def get_traffic_matrix(self):
        """
        Returns the traffic_matrix of this Sample instance.
        """

        return self.traffic_matrix

    def get_srcdst_traffic(self, src, dst):
        """


        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary
            Information stored in the Traffic matrix for the requested src-dst.

        """

        return self.traffic_matrix[src, dst]

    def get_routing_matrix(self):
        """
        Returns the routing_matrix of this Sample instance.
        """

        return self.routing_matrix

    def get_srcdst_routing(self, src, dst):
        """


        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary
            Information stored in the Routing matrix for the requested src-dst.

        """
        return self.routing_matrix[src, dst]

    def get_topology_object(self):
        """
        Returns the topology in networkx format of this Sample instance.
        """

        return self.topology_object

    def get_network_size(self):
        """
        Returns the number of nodes of the topology.
        """
        return self.topology_object.number_of_nodes()

    def get_node_properties(self, id):
        """


        Parameters
        ----------
        id : int
            Node identifier.

        Returns
        -------
        Dictionary with the parameters of the node
        None if node doesn't exist

        """
        res = None

        if id in self.topology_object.nodes:
            res = self.topology_object.nodes[id]

        return res

    def get_link_properties(self, src, dst):
        """


        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary with the parameters of the link
        None if no link exist between src and dst

        """
        res = None

        if dst in self.topology_object[src]:
            res = self.topology_object[src][dst][0]

        return res

    def get_srcdst_link_bandwidth(self, src, dst):
        """


        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Bandwidth in bits/time unit of the link between nodes src-dst or -1 if not connected

        """
        if dst in self.topology_object[src]:
            cap = float(self.topology_object[src][dst][0]['bandwidth'])
        else:
            cap = -1

        return cap


    def _set_data_set_file_name(self,file):
        """
        Sets the data set file from where the sample is extracted.
        """
        self.data_set_file = file

    def _set_performance_matrix(self, m):
        """
        Sets the performance_matrix of this Sample instance.
        """

        self.performance_matrix = m

    def _set_traffic_matrix(self, m):
        """
        Sets the traffic_matrix of this Sample instance.
        """

        self.traffic_matrix = m

    def _set_routing_matrix(self, m):
        """
        Sets the traffic_matrix of this Sample instance.
        """

        self.routing_matrix = m

    def _set_topology_object(self, G):
        """
        Sets the topology_object of this Sample instance.
        """

        self.topology_object = G

    def _set_global_qos_info(self,global_qos_info):
        """
        Sets the global information per QoS of this sample.
        """

        self.global_qos_info = global_qos_info


    def _set_capture_time(self,t):
        """
        Sets the duration of the contemplated experiment in seconds.
        It doesn't considerer the discard time
        """

        self.capture_time = t


    def _set_global_packets(self, x):
        """
        Sets the global_packets of this Sample instance.
        """

        self.global_packets = x

    def _set_global_losses(self, x):
        """
        Sets the global_losses of this Sample instance.
        """

        self.global_losses = x

    def _set_global_delay(self, x):
        """
        Sets the global_delay of this Sample instance.
        """

        self.global_delay = x

    def _get_data_set_file_name(self):
        """
        Gets the data set file from where the sample is extracted.
        """
        return self.data_set_file

    def _get_path_for_srcdst(self, src, dst):
        """
        Returns the path between node src and node dst.
        """

        return self.routing_matrix[src, dst]


    def _get_sizedist_for_srcdst (self, src, dst):
        """
        Returns the size distribution of traffic between node src and node dst.
        """

        return self.traffic_matrix[src, dst]['SizeDist']

    def _get_avgpktsize_for_srcdst_flow (self, src, dst):
        """
        Returns the average packet size for the traffic between node src and
        node dst.
        """

        return self.traffic_matrix[src, dst]['AvgPktSize']

    def _get_sizedistparams_for_srcdst (self, src, dst):
        """
        Returns the time distribution of traffic between node src and node dst.
        """

        return self.traffic_matrix[src, dst]['SizeDistParams']

    def _get_resultdict_for_srcdst (self, src, dst):
        """
        Returns the dictionary with all the information for the communication
        between node src and node dst regarding communication parameters.
        """

        return self.performance_matrix[src, dst]

    def _get_trafficdict_for_srcdst (self, src, dst):
        """
        Returns the dictionary with all the information for the communication
        between node src and node dst regarding size and time distribution
        parameters.
        """

        return self.traffic_matrix[src, dst]

class DatanetAPI:
    """
    Class containing all the functionalities to read the dataset line by line
    by means of an iteratos, and generate a Sample instance with the
    information gathered.
    """

    def __init__ (self, data_folder, shuffle=False):
        """
        Initialization of the PasringTool instance

        Parameters
        ----------
        data_folder : str
            Folder where the dataset is stored.
        shuffle: boolean
            Specify if all files should be shuffled. By default false
        Returns
        -------
        None.

        """

        self.data_folder = data_folder
        self.shuffle = shuffle

        self._all_tuple_files = []
        self._selected_tuple_files = []
        self._graphs_dic = {}
        self._routings_dic = {}
        for root, dirs, files in os.walk(self.data_folder):
            dirs.sort()
            if ("graphs" not in dirs or "routings" not in dirs):
                continue
            files.sort()
            # Extend the list of files to process
            self._all_tuple_files.extend([(root, f) for f in files if f.endswith("tar.gz")])

    def get_available_files(self):
        """
        Get a list of all the dataset files located in the indicated data folder

        Returns
        -------
        Array of tuples where each tuple is (root directory, filename)

        """

        return (self._all_tuple_files.copy())

    def set_files_to_process(self, tuple_files_lst):
        """
        Set the list of files to be processed by the iterator. The files should belong to
        the list of tuples returned by get_available_files.

        Parameters
        ----------
        tuple_files_lst: List of tuples
            List of tuples where each tuple is (path to file, filename)
        """
        if not type(tuple_files_lst) is list:
            raise DatanetException("ERROR: The argument of set_files_to_process should be a list of tuples -> [(root_dir,file),...]")
        for tuple in tuple_files_lst:
            if not type(tuple) is tuple or len(tuple) != 2:
                raise DatanetException("ERROR: The argument of set_files_to_process should be a list of tuples -> [(root_dir,file),...]")
            if (not tuple in self._all_tuple_files):
                raise DatanetException("ERROR: Selected tupla not belong to the list of tuples returned by get_available_files()")

        self._selected_tuple_files = tuple_files_lst.copy()

    def _create_routing_matrix(self, routing_file, net_size):
        """

        Parameters
        ----------

        routing_file : str
            File where the information about routing is located.
        netSize : int
            Number of nodes of the topology

        Returns
        -------
        MatrixPath : NxN Matrix
            Matrix where each cell [i,j] contains the path to go from node
            i to node j.

        """

        MatrixPath = numpy.empty((net_size, net_size), dtype=object)
        with open (routing_file) as fd:
            for line in fd:
                line = line.rstrip()
                nodes = line.split(" ")
                nodes = list(map(int, nodes))
                MatrixPath[nodes[0],nodes[-1]] = nodes

        return (MatrixPath)

    def _generate_graphs_dic(self, path):
        """
        Return a dictionary with networkx objects generated from the GML
        files found in path

        Parameters
        ----------
        path : str
            Direcotory where the graphs files are located.

        Returns
        -------
        Returns a dictionary where keys are the names of GML files found in path
        and the values are the networkx object generated from the GML files.

        """

        graphs_dic = {}
        for topology_file in os.listdir(path):
            G = networkx.read_gml(path+"/"+topology_file, destringizer=int)
            graphs_dic[topology_file] = G

        return graphs_dic

    def __iter__(self):
        """


        Yields
        ------
        s : Sample
            Sample instance containing information about the last line read
            from the dataset.

        """

        g = None

        if (len(self._selected_tuple_files) > 0):
            tuple_files = self._selected_tuple_files
        else:
            tuple_files = self._all_tuple_files

        if self.shuffle:
            random.Random(1234).shuffle(tuple_files)
        ctr = 0
        for root, file in tuple_files:
            if (True):
                it = 0
                tar = tarfile.open(os.path.join(root, file), 'r:gz')
                dir_info = tar.next()
                print(file)
                results_file = tar.extractfile(dir_info.name+"/experimentResults.txt")
                input_files = tar.extractfile(dir_info.name+"/input_files.txt")
                if (dir_info.name+"/QoSexperimentResults.txt" in tar.getnames()):
                    qosresults_file = tar.extractfile(dir_info.name+"/QoSexperimentResults.txt")
                else:
                    qosresults_file = None

                if (dir_info.name+"/traffic_ipg.txt" in tar.getnames()):
                    traffic_ipg_file = tar.extractfile(dir_info.name+"/traffic_ipg.txt")
                else:
                    traffic_ipg_file = None

                while(True):
                    try:
                        s = Sample()
                        s._set_data_set_file_name(os.path.join(root, file))

                        s._results_line = results_file.readline().decode()[:-2]
                        s._input_files_line = input_files.readline().decode()[:-1]
                        if (len(s._results_line) == 0):
                            break

                        fields = s._input_files_line.split(';')
                        num_it = fields[0]
                        s._graph_file = os.path.join(root,"graphs",fields[1])
                        s._routing_file = os.path.join(root,"routings",fields[2])
                        _tm_file = tar.extractfile(dir_info.name+"/tm/tm-"+num_it+".txt")
                        s._tm_file = os.path.join(root,"tm","tm-"+num_it+".txt")
                        if (s._graph_file in self._graphs_dic):
                            g = self._graphs_dic[s._graph_file]
                        else:
                            g = networkx.read_gml(s._graph_file, destringizer=int)
                            self._graphs_dic[s._graph_file] = g


                        # XXX We considerer that all graphs using the same routing file have the same topology
                        if (s._routing_file in self._routings_dic):
                            routing_matrix = self._routings_dic[s._routing_file]
                        else:
                            routing_matrix = self._create_routing_matrix(s._routing_file,len(g))
                            self._routings_dic[s._routing_file] = routing_matrix

                        s._set_routing_matrix(routing_matrix)
                        s._set_topology_object(g)


                        self._process_flow_results(s,_tm_file)
                        if (qosresults_file):
                            s._qos_results_line = qosresults_file.readline().decode()[:-1]
                            self._process_qos_results(s)
                        else:
                            s._qos_results_line = ""

                        if (traffic_ipg_file):
                            s._traffic_ipg_line = traffic_ipg_file.readline().decode()
                            self._process_traffic_ipg_results(s)
                        else:
                            s._traffic_ipg_line = ""

                        it +=1
                        yield s
                    except GeneratorExit :
                        raise
                    except SystemExit :
                        exit(1)
                    except:
                        traceback.print_exc()
                        pass
            '''
            except (GeneratorExit,SystemExit) as e:
                raise
            except:
                #traceback.print_exc()
                print ("Error in the file: %s   iteration: %d" % (file,it))
            '''
            ctr += 1
            #print("Progress check: %d/%d" % (ctr,len(tuple_files)))



    def _process_flow_results(self, s, _tm_file):
        """


        Parameters
        ----------
        s : Sample
            Instance of Sample associated with the current iteration.

        Returns
        -------
        None.

        """

        first_params = s._results_line.split('|')[0].split(',')
        first_params = list(map(float, first_params))
        s._set_capture_time(first_params[0])
        s._set_global_packets(first_params[1])
        s._set_global_losses(first_params[2])
        s._set_global_delay(first_params[3])
        r = s._results_line[s._results_line.find('|')+1:].split(';')

        m_result = []
        m_traffic = []
        for i in range(0,len(r), int(math.sqrt(len(r)))):
            new_result_row = []
            new_traffic_row = []
            for j in range(i, i+int(math.sqrt(len(r)))):
                dict_result_srcdst = {}
                metrics_ = r[j].split(',')
                metrics = list(map(float, metrics_))
                dict_result_agg = {"PktsDrop":metrics[2],
                                   "AvgDelay":metrics[3],
                                   "AvgLnDelay":metrics[4],
                                   "p10Delay":metrics[5],
                                   "p20Delay":metrics[6],
                                   "p50Delay":metrics[7],
                                   "p80Delay":metrics[8],
                                   "p90Delay":metrics[9],
                                   "Jitter":metrics[10]}

                dict_traffic_srcdst = {}
                dict_traffic_agg = {"AvgBw":metrics[0],
                                    "PktsGen":metrics[1],
                                    "TotalPktsGen":metrics[1]*s.capture_time,
                                    "AvgPktSize":metrics[11],
                                    "p10PktSize":metrics[12],
                                    "p20PktSize":metrics[13],
                                    "p50PktSize":metrics[14],
                                    "p80PktSize":metrics[15],
                                    "p90PktSize":metrics[16],
                                    "VarPktSize":metrics[17]}



                dict_result_srcdst['AggInfo'] = dict_result_agg
                dict_traffic_srcdst['AggInfo'] = dict_traffic_agg
                new_result_row.append(dict_result_srcdst)
                new_traffic_row.append(dict_traffic_srcdst)

            m_result.append(new_result_row)
            m_traffic.append(new_traffic_row)
        m_result = numpy.asmatrix(m_result)
        m_traffic = numpy.asmatrix(m_traffic)
        s._set_performance_matrix(m_result)
        s._set_traffic_matrix(m_traffic)
        self._process_traffic_matrix_file(s,_tm_file)



    def _process_traffic_matrix_file(self,s,tm_fd):
        m_traffic = s.traffic_matrix
        line = tm_fd.readline()
        s.max_link_load = float(line)

        for line in tm_fd:
            line = line.decode()
            fields = line.split(";")
            src = int(fields[0])
            dst = int(fields[1])
            flow_list = []
            for flow_params in fields[2:]:
                dict_traffic = {}
                params_lst = list(map(float,flow_params.split(",")))
                if (len(params_lst)== 3):
                    dict_traffic['SizeDist'] = SizeDist.DETERMINISTIC_S
                    size_params = {}
                    size_params['AvgPktSize'] = params_lst[0]
                    dict_traffic['SizeDistParams'] = size_params

                    dict_traffic['TimeDist'] = TimeDist.CBR_T
                    time_params = {}
                    time_params['Rate'] = params_lst[1]
                    dict_traffic['TimeDistParams'] = time_params

                    dict_traffic['ToS'] = int(params_lst[2])
                elif (len(params_lst) == 6):
                    dict_traffic['SizeDist'] = SizeDist.DETERMINISTIC_S
                    size_params = {}
                    size_params['AvgPktSize'] = params_lst[0]
                    dict_traffic['SizeDistParams'] = size_params

                    dict_traffic['TimeDist'] = TimeDist.MULTIBURST_T
                    time_params = {}
                    time_params['On_Rate'] = params_lst[1]
                    time_params['Pkts_per_burst'] = params_lst[2]
                    time_params['IBG'] = params_lst[4]/10**6
                    dict_traffic['TimeDistParams'] = time_params

                    dict_traffic['ToS'] = int(params_lst[5])
                flow_list.append(dict_traffic)

            m_traffic[src,dst]['Flows'] = flow_list


    def _process_qos_results(self,s):
        global_qos_info = {}
        ToS_data_lst = s._qos_results_line.split("/")
        # Init structures:
        for i in range(s.get_network_size()):
            for j in range(s.get_network_size()):
                s.performance_matrix[i,j]["QoS"] = {}
                s.traffic_matrix[i,j]["QoS"] = {}
        for ToS_data in ToS_data_lst:
            idx = ToS_data.find('|')
            global_params_lst = ToS_data[:idx-1].split(",")
            paths_lst = ToS_data[idx+1:].split(";")
            # Process global info

            tos = int(global_params_lst[0])
            global_qos_info[tos] = {"global_packets":float(global_params_lst[2]), \
                "global_losses":float(global_params_lst[3]),"global_delay":float(global_params_lst[4])}
            # Process per path info
            for i in range(s.get_network_size()):
                for j in range(s.get_network_size()):
                    metrics = paths_lst[i*s.get_network_size()+j].split(",")
                    metrics = list(map(float, metrics))
                    qos_perf_dict = {"PktsDrop":metrics[2],
                                   "AvgDelay":metrics[3],
                                   "AvgLnDelay":metrics[4],
                                   "p10Delay":metrics[5],
                                   "p20Delay":metrics[6],
                                   "p50Delay":metrics[7],
                                   "p80Delay":metrics[8],
                                   "p90Delay":metrics[9],
                                   "Jitter":metrics[10]}
                    s.performance_matrix[i,j]["QoS"][tos] = qos_perf_dict
                    qos_traff_dict =  {"AvgBw":metrics[0],
                                    "PktsGen":metrics[1],
                                    "TotalPktsGen":metrics[1]*s.capture_time,
                                    "AvgPktSize":metrics[11],
                                    "p10PktSize":metrics[12],
                                    "p20PktSize":metrics[13],
                                    "p50PktSize":metrics[14],
                                    "p80PktSize":metrics[15],
                                    "p90PktSize":metrics[16],
                                    "VarPktSize":metrics[17]}
                    s.traffic_matrix[i,j]["QoS"][tos] = qos_traff_dict
        s._set_global_qos_info(global_qos_info)


    def _process_traffic_ipg_results(self,s):
        net_size = s.get_network_size()
        m_traffic = s.traffic_matrix
        ipg_per_path_info = s._traffic_ipg_line.split(";")
        for i in range(s.get_network_size()):
            for j in range(s.get_network_size()):
                try:
                    dict_traffic_agg = m_traffic[i,j]['AggInfo']
                    ipg_params_lst = ipg_per_path_info[i*net_size+j].split(",")
                    if (len(ipg_params_lst) == 1):
                        dict_traffic_agg["AvgBurstSize"] = 0
                        dict_traffic_agg["p5BurstSize"] = 0
                        dict_traffic_agg["p95BurstSize"] = 0
                        dict_traffic_agg["varBurstSize"] = 0
                        dict_traffic_agg["AvgOffTime"] = 0
                        dict_traffic_agg["p5OffTime"] = 0
                        dict_traffic_agg["p95OffTime"] = 0
                        dict_traffic_agg["varOffTime"] = 0
                        dict_traffic_agg["meanIPG"] = 0
                        dict_traffic_agg["pIPG_lst"] = [0]*100
                        dict_traffic_agg["varIPG"] = 0
                    else:
                        dict_traffic_agg["AvgBurstSize"] = float(ipg_params_lst[0])
                        dict_traffic_agg["p5BurstSize"] = int(float(ipg_params_lst[1]))
                        dict_traffic_agg["p95BurstSize"] = int(float(ipg_params_lst[2]))
                        dict_traffic_agg["varBurstSize"] = float(ipg_params_lst[3])
                        dict_traffic_agg["AvgOffTime"] = float(ipg_params_lst[4])
                        dict_traffic_agg["p5OffTime"] = float(ipg_params_lst[5])
                        dict_traffic_agg["p95OffTime"] = float(ipg_params_lst[6])
                        dict_traffic_agg["varOffTime"] = float(ipg_params_lst[7])
                        dict_traffic_agg["meanIPG"] = float(ipg_params_lst[8])
                        dict_traffic_agg["pIPG_lst"] = list(map(float,ipg_params_lst[9:-1]))
                        dict_traffic_agg["varIPG"] = float(ipg_params_lst[-1])
                except:
                    traceback.print_exc()
                    exit(1)