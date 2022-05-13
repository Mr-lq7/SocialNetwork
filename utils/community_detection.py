# -*- coding: UTF-8 -*-
import logging
import random
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt


def greedy_modularity_optimization(G):
    """

    :param G: networkx.Graph()
    :return:
    """

    # 最大连通分量
    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc = G.subgraph(largest_cc)

    communities_generator = nx.algorithms.community.greedy_modularity_communities(largest_cc)
    com_list = list(communities_generator)  # [frozenset(), frozenset()], 一个[]就是一个划分

    print(com_list)
    modularity = nx.algorithms.community.modularity(largest_cc, com_list)
    print(f"the modularity score of greedy modularity: {modularity}")
    logging.info(f"the modularity score of greedy modularity: {modularity}")

    # 画图,展示部分节点
    pos = nx.spring_layout(largest_cc)
    plt.figure(figsize=(20, 16))
    nx.draw(largest_cc, pos, with_labels=True, node_size=500, node_color='w', node_shape='o')

    color = ['#000000', '#DCDCDC', '#FFDEAD', '#191970', '#0000CD', '#4682B4', '#00868B', '#006400', '#FFFF00',
             '#EE6363']

    for i in range(len(com_list)):
        random.seed()
        index = random.randint(0, 10)
        nx.draw_networkx_nodes(largest_cc, pos, nodelist=com_list[i], node_size=300, node_color=color[index],
                               node_shape='o')

    plt.title('Greedy Modularity')
    plt.savefig('../result/greedy_modularity.png')
    plt.show()


def girvan_newman(G):
    """
    :param G: networkx.Graph()
    :return:
    """
    # 最大连通分量
    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc = G.subgraph(largest_cc)

    communities_generator = nx.algorithms.community.girvan_newman(largest_cc)
    com_list = list(communities_generator)  # [({}, {}), ({},{}, {})], 一个元组代表一个划分

    for i in com_list:
        modularity = nx.algorithms.community.modularity(largest_cc, list(i))
        print(f"the modularity score of girvan newman in cluster {i}: {modularity}")
        logging.info(f"the modularity score of girvan newman in cluster {i}: {modularity}")

    # 展示部分节点,只展示第0簇信息
    pos = nx.spring_layout(largest_cc)
    plt.figure(figsize=(20, 16))
    nx.draw(largest_cc, pos, with_labels=True, node_size=500, node_color='w', node_shape='o')
    color = ['#000000', '#DCDCDC', '#FFDEAD', '#191970', '#0000CD', '#4682B4', '#00868B', '#006400', '#FFFF00',
             '#EE6363']

    print(com_list[0])
    for i in range(len(com_list[0])):
        random.seed()
        index = random.randint(0, 10)
        nx.draw_networkx_nodes(largest_cc, pos, nodelist=com_list[0][i], node_size=300, node_color=color[index],
                               node_shape='o')

    plt.title('Grivan Newman')
    plt.savefig('../result/grivan_newman.png')
    plt.show()


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


if __name__ == '__main__':
    # 便于分析只用无向图
    logging.basicConfig(filename='community_detection.log', level=logging.INFO)

    # nx_undi_g = networkx_create_undirected_graph('../dataset/Wiki-Vote.txt')
    #
    # greedy_modularity_optimization(G)
    # girvan_newman(nx_undi_g)

    # test

    G = nx.barbell_graph(5, 1)

    greedy_modularity_optimization(G)

    girvan_newman(G)
