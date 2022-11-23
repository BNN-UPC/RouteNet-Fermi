import os, sys
import datanetAPI as api
import random
import numpy as np
from multiprocessing import Pool as ThreadPool


def get_routing_file_name(rname, top_var):
    rname = os.path.basename(rname)

    if (rname.startswith("src_routing")):
        camps = rname.split("_")
        new_rname = "Routing-%s-SrcDst-%s.txt" % (top_var, camps[3])
    else:
        new_rname = rname

    return (new_rname)


def process_routing(sample, routing_file):
    with open(routing_file, "w") as fd:
        R = sample.get_routing_matrix()
        net_size = R.shape[0]
        for i in range(net_size):
            for j in range(net_size):
                if (i == j):
                    continue
                path = ""
                for node in R[i, j]:
                    path += str(node) + ";"
                fd.write(path[:-1] + "\n")


def process_topology(root_path, out_path, samples_per_file, total_samples):
    rpath = os.path.join(out_path, "routings")
    gpath = os.path.join(out_path, "graphs")
    os.mkdir(out_path)
    os.mkdir(rpath)
    os.mkdir(gpath)
    iterators_lst = []

    iterators_lst.append((root_path, iter(api.DatanetAPI(root_path, shuffle=True))))

    j = 0
    i = 0
    res_dir = ""
    t_fd = None
    r_fd = None
    s_fd = None
    i_fd = None
    i0 = 0
    i1 = samples_per_file - 1
    graphs = []
    routings = []

    while (True):
        (path, it) = random.choice(iterators_lst)
        try:
            sample = next(it)
        except StopIteration:
            iterators_lst.remove((path, it))
            if (len(iterators_lst) == 0):
                t_fd.close()
                r_fd.close()
                s_fd.close()
                i_fd.close()
                os.system("tar zcf " + out_path + "/" + res_dir + ".tar.gz -C " + out_path + " " + res_dir)
                os.system("rm -r " + out_path + "/" + res_dir)
                break;
            continue
        if (j == 0):
            if (res_dir != ""):
                t_fd.close()
                r_fd.close()
                s_fd.close()
                i_fd.close()
                os.system("tar zcf " + out_path + "/" + res_dir + ".tar.gz -C " + out_path + " " + res_dir)
                os.system("rm -r " + out_path + "/" + res_dir)
                if (total_samples != -1 and total_samples == i):
                    return

            res_dir = "results_400-2000_%d_%d" % (i0, i1)
            i0 += samples_per_file
            i1 += samples_per_file
            os.mkdir(out_path + "/" + res_dir)
            t_fd = open(out_path + "/" + res_dir + "/traffic.txt", "w")
            r_fd = open(out_path + "/" + res_dir + "/simulationResults.txt", "w")
            s_fd = open(out_path + "/" + res_dir + "/stability.txt", "w")
            i_fd = open(out_path + "/" + res_dir + "/input_files.txt", "w")
            l_fd = open(out_path + "/" + res_dir + "/linkUsage.txt", "w")

        t_fd.write(sample._traffic_line + "\n")
        r_fd.write(sample._results_line + ";\n")
        s_fd.write(sample._status_line + "\n")
        link_usage = sample._link_usage_line
        l_fd.write(link_usage.replace("-nan", "0.000000") + "\n")

        new_graph_file = os.path.basename(sample._graph_file)

        if (not sample._graph_file in graphs):
            graphs.append(sample._graph_file)
            os.system("cp %s %s" % (sample._graph_file, os.path.join(gpath, new_graph_file)))

        new_routing_file = os.path.basename(sample._routing_file)
        if (not sample._routing_file in routings):
            routings.append(sample._routing_file)
            os.system("cp %s %s" % (sample._routing_file, os.path.join(rpath, new_routing_file)))

        i_fd.write("%d;%s;%s\n" % (i, new_graph_file, new_routing_file))
        j += 1
        i += 1
        if (j == samples_per_file):
            j = 0


files = [100, 150, 200, 300]

for i in files:
    print(f"STARTING {i}...")
    root_path = f'../data/scalability/validation2/{i}'
    out_root_dir = f'../data/scalability/validation/{i}'
    samples_per_file = 2
    total_samples = 2

    process_topology(root_path, out_root_dir, samples_per_file, total_samples)
    print("FINISHING...")