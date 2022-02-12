"""
Assignment 01
24.01.2022
author: Annabell Kießler, Davide ...
"""

# imports
import time
import numpy as np
import re
import matplotlib.pyplot as plt
import scipy as sc
from scipy.sparse import csr_matrix
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree


def read_coordinate_file(filename):
    """
    reads coordinates from textfile and converts them into Mercator projection

    :param filename:
    :return: 2D np_array of coordinates
    """

    line_list = []
    with open(filename, 'r') as file :  # endet mit leerem array!

        line = file.readline()

        while line:
            split_text = re.split("{|}|,", line)  # result: ['', ' a ', ' b ', '\n']
            a_n = float((split_text[1].rsplit())[0])
            b_n = float((split_text[2].rsplit())[0])

            x_n = (np.pi * b_n) / 180
            y_n = np.log(np.tan(np.pi / 4 + (np.pi * a_n) / 360))

            line_list.append((x_n, y_n))

            line = file.readline()

    return np.array(line_list)

def construct_fast_graph_connections(coord_list, radius):
    ind_list = []
    dist_list = []
    mytree = cKDTree(coord_list)
    ind = mytree.query_ball_point(coord_list, radius)
    print(ind[0])
    for i in ind:

        for j in i:
            print(coord_list[i[0]])
            print(coord_list[j])
            dist_list.append(sc.spatial.distance.cdist(coord_list[i[0]], coord_list[j]))
            ind_list.append((i[0], j))



    ind_array = np.array(ind)
    dist_array = np.array(dist_list)
    #print(dist_array)

    return ind_array, dist_array


def construct_graph_connections(coord_list, radius):
    """

    calculates the distance between the cities and register the pairs who are in the given radius
    :param coord_list:
    :param radius:
    :return: 2Darray with the indices of the connected cities, 1Darray with the distances
    """
    outputIndices = []
    outputDistance = []

    for indiceI, i in enumerate(coord_list):
        indiceJ = indiceI+1
        while indiceJ < len(coord_list):
            distance = np.sqrt(np.square(coord_list[indiceJ, 0] - i[0]) + np.square(coord_list[indiceJ, 1] - i[1]))
            if 0 < distance < radius:
                outputIndices.append((indiceI, indiceJ))
                outputDistance.append(distance)
            indiceJ = indiceJ+1


    return np.array(outputIndices), np.array(outputDistance)

def construct_graph(indices, distance, N):
    """
    creates a matrix of the connections
    :param indices:
    :param distance:
    :param N:
    :return: Matrix where the indice of the row equals the indice of the first city and the indice of the coloumn equals the second city. The element in this slot is the distance between them.
    """
    #np.set_printoptions(threshold=np.inf) #zeigt ganze matrix an
    output = csr_matrix((distance, (indices[:, 0], indices[:, 1])), shape=(N, N)).toarray()

    return output

def find_shortest_path(graph, start_node, end_node): #hungary: start:311 end:702
    """
    finds the shortest path from the start city to the end city with connecting cites in the radius on its way
    :param graph: csr matrix
    :param start_node: indice start city
    :param end_node: indice end city
    :return: array with indices the cities on the shortest path in the order from start_node to end_node
    """
    dist_matrix, predecessors = sc.sparse.csgraph.shortest_path(graph, method='auto', directed=False, return_predecessors=True, indices=end_node)
    distance_start_end = dist_matrix[start_node]

    path_array = np.array(start_node)
    indice_p = predecessors[start_node]
    i = 0

    while indice_p != end_node and i < 100:
        path_array = np.append(path_array, indice_p)
        indice_p = predecessors[indice_p]
        i = i+1

    path_array = np.append(path_array, end_node)
    return path_array, distance_start_end

def plot_points(coord_list, indices, path):
    """
    plots the points of the coordinates list on a graph
    :param coord_list: the coordinates
    :return: a graph of the plotted points
    """

    print(indices)
    connections = []
    for i in indices:

        connection_i = []

        for j in i:
            connection_i.append((coord_list[j, 0],coord_list[j, 1]))

        connections.append(connection_i)

    lines = LineCollection(connections, linewidths=0.15, color="grey")
    fig, ax = plt.subplots()
    ax.set_xlim(0.28, 0.4) #muss für deutschland angepasst werden
    ax.set_ylim(0.9, 0.97) #anpassen!
    ax.add_collection(lines)

    xy2 = zip(*coord_list)
    ax.scatter(*xy2, s=10, color = "red")

    x_values = []
    y_values = []
    for i in path:
        x_values.append(coord_list[i, 0])
        y_values.append(coord_list[i, 1])
    ax.plot(x_values,y_values, color="blue")

    return plt.show()

#data
file = "HungaryCities.txt"
radius = 0.005
start_node = 311 #1573
end_node = 702 #10584


#reads coordinate file
start_time = time.time()
array_cities = read_coordinate_file(file)
print("read_coordinate_file(): %s seconds" % (time.time() - start_time))

#construct graph fast
indices, distances = construct_fast_graph_connections(array_cities, radius)

# constructs graph connections
#start_time = time.time()
#indices, distances = construct_graph_connections(array_cities, radius)
#print("construct_graph(): %s seconds" % (time.time() - start_time))


#csr-matrix erstellen
#graph = construct_graph(indices, distances, max(np.shape(array_cities)))

#shortest path
#path, dist = find_shortest_path(graph, start_node, end_node)

#plot map
#plot_points(array_cities, indices, path)

"""
def plot_points(coord_list):

    plots the points of the coordinates list on a graph
    :param coord_list: the coordinates
    :return: a graph of the plotted points

    xy2=zip(*coord_list)
    plt.plot(*xy2,'o')
    plt.yscale("log")
    return plt.show()
    
    def plot_points(coord_list, indices) :

    :param coord_list: 2D np_array of coordinates (output of read_coordinate_files)
    :param indices: 2D np_array with the indices of the connected cities (output of construct_graph_connections)
    :return: image of the maps with the cities and the connections
    
    xy2 = zip(*coord_list)

    for i in indices:
        pass

    plt.plot(*xy2,"o")

    fig, ax = plt.subplots()
    for idx_pair in indices:
        start = coord_list[idx_pair[0]]
        end = coord_list[idx_pair[1]]
        arr = np.stack((start, end), axis=0)

        l = LineCollection(arr[:, 0], arr[:, 1], "-")
        ax.add_collection(l)
    plt.yscale("log")
    return plt.show()
"""

