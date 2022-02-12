"""
Assignment 01
06.02.2022
author: Annabell Kießler, Davide Alpino
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

#functions
def read_coordinate_file(filename):
    """
    reads coordinates from textfile and converts them into Mercator projection

    :param filename: file with coordinates(x,y)
    :return: 2D np_array of coordinates
    """

    line_list = []  # list that will contain the transformed coordinates in the end
    with open(filename, 'r') as file :

        line = file.readline()

        while line:
            split_text = re.split("{|}|,", line)  # splits line in ['', ' a ', ' b ', '\n']
            a_n = float((split_text[1].rsplit())[0])  # separates x-coordinate
            b_n = float((split_text[2].rsplit())[0])  # separates y-coordinate

            x_n = (np.pi * b_n) / 180  # calculates x in mercator projection
            y_n = np.log(np.tan(np.pi / 4 + (np.pi * a_n) / 360))  # calculates y in mercator projection

            line_list.append((x_n, y_n))

            line = file.readline()

    return np.array(line_list)


def construct_graph_connections(coord_list, radius):
    """
    calculates the distance between the cities and register the pairs who are in the given radius

    :param coord_list: 2D np_array of coordinates
    :param radius: int
    :return: 2D np_array with the indices of the connected cities, 1D np_array with the distances between these cities
    """
    output_indices = []
    output_distance = []

    for indice_city_i, coord_city_i in enumerate(coord_list):  # loop 1: select city i: iterate over all cities
        indice_city_j = indice_city_i+1
        while indice_city_j < len(coord_list):  # loop 2: select city j: iterate over all cities that loop1 haven't checked jet
            distance = np.sqrt(np.square(coord_list[indice_city_j, 0] - coord_city_i[0]) + np.square(coord_list[indice_city_j, 1] - coord_city_i[1]))  # calculates distance between city 1 and city 2
            if distance < radius:
                output_indices.append((indice_city_i, indice_city_j))  # appends Indices of the city if they are in the radius
                output_distance.append(distance)
            indice_city_j = indice_city_j+1  # iteration for while loop

    return np.array(output_indices), np.array(output_distance)


def construct_fast_graph_connections(coord_list, radius):
    """
    calculates the distance between the cities and register the pairs who are in the given radius (fast)

    :param coord_list: 2D np_array of coordinates
    :param radius: int
    :return: 2D np_array with the indices of the connected cities, 1D np_array with the distances between these cities
    """
    kdTree = cKDTree(coord_list)
    indices_radius = kdTree.query_ball_point(coord_list, radius)
    indices = []
    distance = []
    k = 0
    while k < len(indices_radius):  # pointer of the line in indices_radius corresponds to Indices_city1
        for i in indices_radius[k]:  # iterate over every element in line corresponds to Indices_city2
            if k != i:  # to eliminate city1 = city2
                indices.append((k, i))
                distance.append(np.linalg.norm(coord_list[k] - coord_list[i]))
        k += 1  # iteration for while loop

    return np.array(indices), np.array(distance)


def construct_graph(indices, distance, n):
    """
    creates a matrix of the connections

    :param indices: 2D np_array with the indices of the connected cities
    :param distance: 1D np_array with the distances between these cities
    :param n: shape of matrix = number of cities
    :return: matrix where the indice of the row equals the indice of the first city and the indice of the coloumn equals the second city. The element in this slot is the distance between them.
    """

    output = csr_matrix((distance, (indices[:, 0], indices[:, 1])), shape=(n, n)).toarray()

    return output


def find_shortest_path(graph, start_node, end_node): #hungary: start:311 end:702
    """
    finds the shortest path from the start city to the end city with connecting cites in the radius on its way

    :param graph: csr matrix
    :param start_node: indice start city
    :param end_node: indice end city
    :return: 1D np_array with indices of the cities on the shortest path in the order: start_node to end_node
    """
    dist_matrix, predecessors = sc.sparse.csgraph.shortest_path(graph, method='auto', directed=False, return_predecessors=True, indices=end_node)  # calculates distance from the end_city to all other cities
    distance_start_end = dist_matrix[start_node]  # distance from start_city to the end_city

    path_array = np.array(start_node)  # output_array that will contain the Indices of the cities on the path in order: start to end
    indice_p = predecessors[start_node]  # indices of the predecessor of the start_city
    i = 0

    while indice_p != end_node and i < 10000:  # goes through loop until the end_city = predecessor, i --> to prevent infinity while loops
        path_array = np.append(path_array, indice_p)  # appends predecessor to output_array
        indice_p = predecessors[indice_p]  # takes predecessor of the current indices_p
        i = i+1

    path_array = np.append(path_array, end_node) # adds end_city to path

    return path_array, distance_start_end


def plot_points(coord_list, indices, path):
    """
    plots the points of the coordinates list, the connections between the cities in the radius, the shortest path

    :param coord_list: 2D np_array of coordinates
    :param indices: 2D np_array with the indices of the connected cities
    :param path: 1D np_array with indices of the cities on the shortest path
    :return: plot
    """

    #connections
    connections = []  # will contain every connection with there coordinates((x_city1,y_city1),(x_city_2,y_city2))

    for i in indices:  # iterates through every line of indices

        connection_i = []  # reset current line (connection)

        for j in i:  # takes city1/2 of current line
            connection_i.append((coord_list[j, 0],coord_list[j, 1]))  # takes coordinates of current city_1 and city_2

        connections.append(connection_i)  # appends current connection to list of all connections

    lines = LineCollection(connections, linewidths=0.2, color="grey")
    fig, ax = plt.subplots()
    ax.add_collection(lines)

    #cities
    xy2 = zip(*coord_list)
    ax.scatter(*xy2, s=10, color="red")

    #shortest path
    x_values = []
    y_values = []
    for i in path:
        x_values.append(coord_list[i, 0])
        y_values.append(coord_list[i, 1])
    ax.plot(x_values,y_values, color="blue")

    return plt


###########################################################################################
#data
file = "SampleCoordinates.txt"
radius = 0.08
start_node = 0 #1573 #311
end_node = 5 #10584 #702

#reads coordinate file
start_time = time.time()
array_cities = read_coordinate_file(file)
print("time: read_coordinate_file(): %s seconds" % (time.time() - start_time))

#construct graph connections !fast!
#start_time = time.time()
#indices, distances = construct_fast_graph_connections(array_cities, radius)
#print("time: construct fast_graph_connections(): %s seconds" % (time.time() - start_time))

# constructs graph connections
start_time = time.time()
indices, distances = construct_graph_connections(array_cities, radius)
print("time: construct_graph_connections(): %s seconds" % (time.time() - start_time))

#csr-matrix erstellen
start_time = time.time()
graph = construct_graph(indices, distances, max(np.shape(array_cities)))
print("time: construct_graph(): %s seconds" % (time.time() - start_time))

#shortest path
start_time = time.time()
path, dist = find_shortest_path(graph, start_node, end_node)
print("shortest path:", path)
print("total distance:", dist)
print("time: find_shortest_path(): %s seconds" % (time.time() - start_time))

#plot map
start_time = time.time()
plot = plot_points(array_cities, indices, path)
print("time: plot_points(): %s seconds" % (time.time() - start_time))
plot.show()
