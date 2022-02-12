# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from matplotlib.collections import LineCollection


# Import coordinates of the cities
def read_coordinate_file(filename):
    """
    read_coordinate_file(''filename''):
    Imports a file with longitude and latitude in [°].
    Then the data gets processed to output coordinates in x, y Format using 'Mercator projection'.
    The output type is a NumPy 2 dimensional array.
    """
    R = 1  # Normalized Radius for Mercator projection

    with open(filename, mode='r') as input_file:
        lines = input_file.readlines()
        export = np.empty((len(lines), 2))  # Empty Array with same size like data in input file
        i_help = 0
        for line in lines:
            line_replace = line.replace('{', '').replace('}', '').rstrip('\n')
            line_replace_split = line_replace.split(sep=',')
            # Calculate coordinates by using Mercator projection
            export[i_help][0] = R * np.pi / 180 * float(line_replace_split[1])
            export[i_help][1] = R * np.log(np.tan(np.pi / 4 + np.pi / 360 * float(line_replace_split[0])))
            i_help = i_help + 1

    return export


# Plot the cities and connections which are inside the radius
def plot_points(coord_list, indices):
    # Calculate the connection lines between the cities
    lines = []

    for index in indices:
        line = []
        for i in index:
            line.append((coord_list[i, 0], coord_list[i, 1]))

        lines.append(line)

    line_segments = LineCollection(lines, linewidths=0.2, color = '0.4')

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(coord_list[:, 0], coord_list[:, 1], s=10, color = 'red')
    # plt.xlim(0.05, 0.35)
    ax.add_collection(line_segments)
    fig.savefig('cities.svg')
    fig.show()



# Calculate which cities can be connected (within the radius)
def construct_graph_connections(coord_list, radius):
    index_array = np.zeros((1,2),dtype=int)
    distance_output = np.zeros(1)
    number_of_cities = max(np.shape(coord_list))
    for index_city1, coord_city1 in enumerate(coord_list):
        for index_city2, coord_city2 in enumerate(coord_list):
            x_diff = coord_city1[0] - coord_city2[0]
            y_diff = coord_city1[1] - coord_city2[1]
            distance = np.sqrt(x_diff ** 2 + y_diff ** 2)
            if distance <= radius and distance > 0:
                index_array = np.append(index_array, [[index_city1, index_city2]], axis=0)
                distance_output = np.append(distance_output, distance)
    index_array = np.delete(index_array, 0, axis=0)
    distance_output = np.delete(distance_output, 0)

    return index_array, distance_output


# Create sparse matrix with the distance between the cities
def construct_graph(indices, distance , N):
    output_matrix = csr_matrix((distance, (indices[:, 0], indices[:, 1])), shape=(N, N)).toarray()
    return output_matrix


# Call the function: Task 1
#test_coordinates = read_coordinate_file('SampleCoordinates.txt')
#r_range = 0.08

# test_coordinates = read_coordinate_file('GermanyCities.txt')
# r_range = 0.0025

test_coordinates = read_coordinate_file('HungaryCities.txt')
r_range = 0.005



# Call the function: Task 3
indices, distance = construct_graph_connections(test_coordinates, r_range)
print(indices,distance)
# Call the function: Task 4
#print(construct_graph(indices, distance, max(np.shape(test_coordinates))))

# Call the function: Task 2
#plot_points(test_coordinates, indices)