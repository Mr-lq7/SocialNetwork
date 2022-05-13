# -*- coding: UTF-8 -*-
import logging
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import powerlaw




def networkx_create_directed_graph(filepath):
    """
    :param filepath: file path
    :return: DiGraph
    """

    G = nx.DiGraph()
    with open(filepath) as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.startswith('#') is False]
    edges = [(int(line.split()[0]), int(line.split()[1])) for line in lines]
    G.add_edges_from(edges)
    return G


def networkx_create_undirected_graph(filepath):
    """
    :param filepath: file path
    :return: Graph
    """

    G = nx.Graph()
    with open(filepath) as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.startswith('#') is False]
    edges = [(int(line.split()[0]), int(line.split()[1])) for line in lines]
    G.add_edges_from(edges)
    return G

def plot_degree_distribution(G):

    degs = defaultdict(int)
    for k, v in G.degree():
        degs[v] += 1

    # 按照key的升序排序
    items = sorted(degs.items())
    # x: node y: frequency
    x, y = np.array(items).T
    # y_sum = np.sum(y)
    # y = [float(i)/y_sum for i in y]
    plt.plot(x, y, 'b-o')
    plt.legend(['Degree'])
    plt.xlabel('$K$', fontsize=20)
    plt.ylabel('$P_K$', fontsize=20)
    plt.title('$Degree\ Distribution$', fontsize=20)
    plt.savefig('../result/degree_distribution.png')
    plt.show()

def plot_loglog_degree_distribution(G):
    degs = defaultdict(int)
    for k, v in G.degree():
        degs[v] += 1

    items = sorted(degs.items())

    x, y = np.array(items).T
    plt.plot(x, y, 'b-o')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['Degree'])
    plt.xlabel('$K$', fontsize=20)
    plt.ylabel('$P(K)$', fontsize=20)
    plt.title('$Log\ Log\ Degree\ Distribution$', fontsize=20)
    plt.savefig('../result/loglog_degree_distribution.png')
    plt.show()

def plot_clustering_coefficient_distribution(clustering_coefficient):
    ccs = defaultdict(int)
    for k, v in clustering_coefficient.items():
        ccs[v] += 1

    items = sorted(ccs.items())
    x, y = np.array(items).T
    plt.plot(x, y, 'b-o')
    plt.legend(['Clustering Coefficient Distribution'])
    plt.xlabel('$Clustering Coefficient, C$', fontsize=20)
    plt.ylabel('$Frequency, F$', fontsize=20)
    plt.title('$Clustering Coefficient Distribution$', fontsize=20)
    plt.savefig('../result/clustering_coefficient_distribution.png')
    plt.show()

def plot_coreness_distribution(coreness):
    cores = defaultdict(int)
    for k, v in coreness.items():
        cores[v] += 1

    items = sorted(cores.items())
    x, y = np.array(items).T
    plt.plot(x, y, 'b-o')
    plt.legend(['Coreness Distribution Distribution'])
    plt.xlabel('$Coreness Distribution$', fontsize=20)
    plt.ylabel('$Frequency, F$', fontsize=20)
    plt.title('$Coreness Distribution Distribution$', fontsize=20)
    plt.savefig('../result/coreness_distribution.png')
    plt.show()


if __name__ == '__main__':

    logging.basicConfig(filename='network statistics.log', level=logging.INFO)

    nx_di_g = networkx_create_directed_graph('../dataset/Wiki-Vote.txt')

    print(f"networkx_create_directed_graph #Nodes: {nx_di_g.number_of_nodes()}")
    print(f"networkx_create_directed_graph #Edges: {nx_di_g.number_of_edges()}")
    logging.info(f"networkx_create_directed_graph #Nodes: {nx_di_g.number_of_nodes()}")
    logging.info(f"networkx_create_directed_graph #Edges: {nx_di_g.number_of_edges()}")

    nx_undi_g = networkx_create_undirected_graph('../dataset/Wiki-Vote.txt')

    print(f"networkx_create_undirected_graph #Nodes: {nx_undi_g.number_of_nodes()}")
    print(f"networkx_create_undirected_graph #Edges: {nx_undi_g.number_of_edges()}")
    logging.info(f"networkx_create_undirected_graph #Nodes: {nx_undi_g.number_of_nodes()}")
    logging.info(f"networkx_create_undirected_graph #Edges: {nx_undi_g.number_of_edges()}")

    # self_loop
    print(f"undirected graph #number of self_loop: {nx.number_of_selfloops(nx_undi_g)}")
    print(f"directed graph #number of self_loop: {nx.number_of_selfloops(nx_di_g)}")
    logging.info(f"undirected graph #number of self_loop: {nx.number_of_selfloops(nx_undi_g)}")
    logging.info(f"directed graph #number of self_loop: {nx.number_of_selfloops(nx_di_g)}")

    nx_di_g.remove_edges_from(nx.selfloop_edges(nx_di_g))
    nx_undi_g.remove_edges_from(nx.selfloop_edges(nx_undi_g))

    # 便于分析只用无向图

    # 连通分量的个数
    components_num = nx.number_connected_components(nx_undi_g)
    print(f"#number of connected components: {components_num}")
    logging.info(f"#number of connected components: {components_num}")

    # 最大连通分量
    largest_cc = max(nx.connected_components(nx_undi_g), key=len)
    largest_cc = nx_undi_g.subgraph(largest_cc)
    print(f"#node number of largest connected components: {largest_cc.number_of_nodes()}")
    print(f"#edge number of largest connected components: {largest_cc.number_of_edges()}")
    logging.info(f"#node number of largest connected components: {largest_cc.number_of_nodes()}")
    logging.info(f"#edge number of largest connected components: {largest_cc.number_of_edges()}")

    # 度分布
    degree_distribution = nx.degree_histogram(nx_undi_g)
    print(f"all nodes' degree: {degree_distribution}")
    logging.info(f"all nodes' degree: {degree_distribution}")

    plot_degree_distribution(nx_undi_g)
    plot_loglog_degree_distribution(nx_undi_g)

    # 平均度
    degrees = [v for k, v in nx_undi_g.degree()]
    print(f"maximun degree: {np.max(degrees)}")
    print(f"average degree: {np.mean(degrees)}")
    logging.info(f"maximum degree: {np.mean(degrees)}")
    logging.info(f"average degree: {np.mean(degrees)}")

    # 参数估计
    data = np.array(degrees)  # data can be list or numpy array
    results = powerlaw.Fit(data)
    print(f"alpha: {results.power_law.alpha}")
    print(f"xmin: {results.power_law.xmin}")
    print(f"fit_method: {results.fit_method}")
    logging.info(f"alpha: {results.power_law.alpha}")
    logging.info(f"xmin: {results.power_law.xmin}")
    logging.info(f"fit_method: {results.fit_method}")

    # 最大连通分量的直径
    diameter = nx.diameter(largest_cc)
    print(f"graph diameter: {diameter}")
    logging.info(f"graph diameter: {diameter}")

    # 最大连通分量的平均最短路径长度
    average_shortest_path_length = nx.average_shortest_path_length(largest_cc)
    print(f"graph average shortest path length: {average_shortest_path_length}")
    logging.info(f"graph average shortest path length: {average_shortest_path_length}")

    # 聚集系数
    clustering_coefficient = nx.clustering(nx_undi_g)
    # plot_clustering_coefficient_distribution(clustering_coefficient)
    ccs = [v for k, v in clustering_coefficient.items()]
    plt.hist(ccs)
    plt.xlabel("$Clustering Coefficient, C$")
    plt.ylabel("$Frequency, F$")
    plt.title("Clustering Coefficient Distribution")
    plt.savefig("../result/clustering_coefficient_distribution.png")
    plt.show()

    # 平均聚集系数
    avg_cc = nx.average_clustering(nx_undi_g)
    print(f"graph average clustering coefficient: {avg_cc}")
    logging.info(f"graph average clustering coefficient: {avg_cc}")


    # 核数
    coreness = nx.algorithms.core.core_number(nx_undi_g)
    # plot_coreness_distribution(coreness)
    cores = [v for k, v in coreness.items()]

    plt.hist(cores)
    plt.xlabel("$Coreness\ Distribution$", fontsize=20)
    plt.ylabel("$Frequency, F$", fontsize=20)
    plt.title("$Coreness\ Distribution$", fontsize=20)
    plt.savefig('../result/coreness_distribution.png')
    plt.show()

    # nx.draw(nx_undi_g)
    # plt.show()
    # plt.xlabel("$Clustering Coefficient, C$")
    # plt.ylabel("$Frequency, F$")
    # plt.title("Clustering Coefficient Distribution")
    # plt.savefig("../result/clustering_coefficient_distribution.png")
    # plt.show()
    # print(components_num)
    # g = nx.Graph()
    # g.add_edge(1,1)
    # g.add_edge(1,2)
    # print(nx.number_of_selfloops(g))

    #
    # print([i for i in nx.nodes_with_selfloops(g)])
    # g.remove_edges_from(nx.selfloop_edges(g))
    # print(g.number_of_edges())
    # print(g.number_of_nodes())
