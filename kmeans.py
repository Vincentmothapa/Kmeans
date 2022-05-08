'''
ALGORITHM

Ask user for the file to read data from.

Call function to extract file data and assign it to the relevant
variables for manipulation.

Ask user for number of clusters to group data into.

Call function to find clusters and return the center for each cluster
and a label list showing which cluster each data point belongs to.

From labels, find how many countries are in each cluster and print.

Find the names of each country in each cluster.

Find the average birthrate and life expectancy per cluster.

Plot the clustered data in different colours.
'''


import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from collections import Counter, defaultdict


# ========================================================================
# Define a function that reads data in from the csv files and returns
# data to be used for the rest of the code.
# HINT: http://docs.python.org/2/library/csv.html. 

'''
This function reads the .csv file chosen by the user. The headings for the columns are
saved on x_label and y_label to be used for the x and y labels when plotting. List x is
saves the birthrate and y saves the life expectancy. x and y are combined to form a 2D
list xy for easier manipulation going forward
'''

def readCSV(file):
    # Create empty lists to add information to
    x = []
    y = []
    country = []
    x_label = None
    y_label = None
    
    with open(file) as csvfile:
        read = csv.reader(csvfile, delimiter = ',')
        temp = 0

        for row in read:
            # This if statement is to store the headings separately and the data separately
            if temp == 1:
                country.append(row[0])
                x.append(float(row[1]))
                y.append(float(row[2]))
            else:
                x_label = row[1]
                y_label = row[2]
                temp = 1
    
    # Combine x and y to form cordinates for easier manipulation
    xy = np.vstack((x, y)).T
    

    # The code below was used to plot the initial data to see how it looks
    '''
    # Plot the initial data points before clustering
    plt.title("Data before clustering")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y, color='r')
    plt.show()
    '''
    
    return x_label, y_label, xy, country


#========================================================================
# Write a function to visualise the clusters.
# (optional, but useful to see the changes and if your algorithm is working)

def plot_clusters(x_label, y_label, xy, cluster_num, centers, labels):
    plt.scatter(xy[:, 0], xy[:, 1], c = labels, s = 50, cmap = 'cividis')
    plt.title('K-Means clustering of countries by birth rate vs life expectancy')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    
#========================================================================
# Write the initialisation procedure

'''
I got help with this function from Aden Haussmann through his article in Towards
Data Science titled "K-Means Clustering for Beginners" and taylored the solution
to this specific task.
https://towardsdatascience.com/k-means-clustering-for-beginners-ea2256154109

The function takes in the xy list as defined under readCVS() and the number of clusters
from the user and returns cluster centers and labels describing which cluster each data point
in xy lies.
'''

def initClusters(xy, cluster_num):
    # Randomly choose cluster index
    rng = np.random.RandomState(2)
    i = rng.permutation(xy.shape[0])[:cluster_num]
    centers = xy[i]
    
    control = 0
    iterations = int(input("Number of iterations: "))
    while control < iterations:
        control = control + 1
        labels = pairwise_distances_argmin(xy, centers)        
        new_centers = np.array([xy[labels == i].mean(0) for i in range(cluster_num)])
        
    return new_centers, labels


def main():
    # ========================================================================
    # Ask the user what dataset they'd like to use.
    # Declare variables and assign them the returned values of the function readCSV

    print("Available information:\na. 1953 data\nb. 2008 data\nc. data for both years")

    data_option = input("Enter the letter corresponding to the data of interest: ")
    data_set = ""
    if data_option == 'a':
        data_set = 'data1953.csv'
    elif data_option == 'b':
        data_set = 'data2008.csv'
    elif data_option == 'c':
        data_set = 'dataBoth.csv'
    else:
        print("Invalid data option")
        sys.exit()
      
    x_label, y_label, xy, country = readCSV(data_set)


    # ========================================================================
    # Initalising cluster points

    cluster_num = int(input("Input number of clusters: "))                     
    centers, labels = initClusters(xy, cluster_num)


    # ========================================================================
    # Print out the results for questions
    #1.) The number of countries belonging to each cluster
    print("\nCountries in each cluster: ")
    print(Counter(labels))


    #2.) The list of countries belonging to each cluster
    #3.) The mean Life Expectancy and Birth Rate for each cluster
    print("\nCountries according to their clusters")

    # Get cluster indices first
    indices = defaultdict(list)
    for index, c in enumerate(labels):
        indices[c].append(index)

    control = 0
    while control < cluster_num:
        print("\nCluster " + str(control + 1) + ":")
        for i in indices[control]:
            print(country[i])
            
        print("\nAverage birth rate:")
        print(centers[control][0])
        print("Average life expectancy:")
        print(centers[control][1])

        control = control + 1

          
    # ========================================================================
    # Plot clusters
    # Plot clusters is called at the very end of the program to allow for other
    # information to be printed first.
    plot_clusters(x_label, y_label, xy, cluster_num, centers, labels)

main()


'''
REFERENCES
1. Finding index of minimum value in array
https://thispointer.com/numpy-amin-find-minimum-value-in-numpy-array-and-its-index/

2. I learnt about combining 2 lists into one 2D list from Stackoverflow
https://stackoverflow.com/questions/53074230/how-to-combine-list-of-x-and-list-of-y-into-one-x-y-list

3. matplotlib colourmaps to color the different clusters
https://matplotlib.org/stable/tutorials/colors/colormaps.html


'''
