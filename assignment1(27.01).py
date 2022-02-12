"""
Assignment 01
24.01.2022
author: Annabell Kießler, Davide ...
"""

# imports
import numpy as np
import re
import matplotlib.pyplot as plt
import scipy as sc


# functions
def mercator_projection_x(b) :
    """
    calculates mercator_projection of x

    :param b:
    :return:
    """
    x = (np.pi * b) / 180
    return x


def mercator_projection_y(a) :
    """
    calculates mercator_projection of y

    :param a:
    :return:
    """
    y = np.log(np.tan(np.pi / 4 + (np.pi * a) / 360))
    return y


def read_coordinate_file(filename) :
    """
    reads coordinates from textfile and converts them into Mercator projection

    :param filename:
    :return:
    """

    with open(filename, 'r') as file :  # endet mit leerem array!
        split_text = re.split("{|}|,", file.readline())  # ['', ' a ', ' b ', '\n']
        coord_array = np.array([[mercator_projection_x(float((split_text[2].rsplit())[0])), mercator_projection_y(
            float((split_text[1].rsplit())[0]))]])  # initialisierung des ersten elements vom array

        line = file.readline()  # zweite line

        while line :
            split_text = re.split("{|}|,", line)  # ['', ' a ', ' b ', '\n']
            a_n = float((split_text[1].rsplit())[0])
            b_n = float((split_text[2].rsplit())[0])

            x_n = mercator_projection_x(b_n)
            y_n = mercator_projection_y(a_n)
            coord_array = np.append(coord_array, [[x_n, y_n]], axis=0)

            line = file.readline()  # liest neue Zeile für nächste interation aus

    #N = np.shape(coord_array)[0] #number of cities --> muss irgendwie übergeben werden

    return coord_array


def plot_points(coord_list):
    """
    plots the points of the coordinates list on a graph
    :param coord_list: the coordinates
    :return: a graph of the plotted points
    """
    xy2=zip(*coord_list)
    plt.plot(*xy2,'o')
    plt.yscale("log")
    return plt.show()

def construct_graph_connections(coord_list, radius):
    """

    calculates the distance between the cities and register the pairs who are in the given radius
    :param coord_list:
    :param radius:
    :return: 2Darray with the indices of the connected cities, 1Darray with the distances
    """
    outputIndices = np.array([[0, 0]]) #create array
    outputDistance = np.array([[0]]) #create array

    for indiceI, i in enumerate(coord_list):
        for indiceJ, j in enumerate(coord_list):
            if 0 < np.linalg.norm(i - j) < radius:
                outputIndices = np.append(outputIndices, [[indiceI, indiceJ]], axis=0)
                outputDistance = np.append(outputDistance, [[np.linalg.norm(i - j)]], axis=0)

    outputIndices = np.delete(outputIndices, obj=0, axis=0)# first element gets deleted [0,0]
    outputDistance = np.delete(outputDistance,obj=0,axis=0) # first element gets deleted [0]

    return outputIndices, outputDistance

def construct_graph(indices, distance, N): # N = Anzahl Städte wie übergeben???
    """
    creates a matrix of the connections
    :param indices:
    :param distance:
    :param N:
    :return: Matrix where the indice of the row equals the indice of the first city and the indice of the coloumn equals the second city. The element in this slot is the distance between them.
    """
    output = sc.csr_matrix((distance, indices), shape=(N, N))

    return output

#print(read_coordinate_file("HungaryCities.txt"))
#plot_points(read_coordinate_file("HungaryCities.txt")[0])
#construct_graph(construct_graph_connections(read_coordinate_file("HungaryCities.txt"), 0.01), 850) #wie übergebe ich mehrere outputs?
