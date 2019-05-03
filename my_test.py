''' 
"Generating DTW Means of Time Series for Data Visualization"
CS2270: Final Project 
Author: jweissko
Date: 04/28/2019
'''
# Download and install these libraries if you don't have them installed already
from collections import Counter
import datetime
# the download for dtaidistance can be found here: https://github.com/wannesm/dtaidistance
from dtaidistance import dtw
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.interpolate as interp

# function to read csv file, obtain the standard sereis length, and get indices of the standard series starting positions
def read_data(csv_file_name, discriminatory_column_name):
    # read csv file
    csvfile = pd.read_csv(csv_file_name)
    # column of stock names. Note: You can also use csvfile['Name']
    discriminatory_column = csvfile[discriminatory_column_name]
    return csvfile, discriminatory_column

csvfile, name_column = read_data('all_stocks_5yr.csv', 'Name')

# function to get the standard size of a series, the number of standard sized series, and the starting indices of such series
def standard_length(csvfile, discriminatory_column):
    # Get idxs where the stock name changes adn via that series sizes. Start with idx 0
    change_idxs = [0]
    series_sizes = []
    for i in range(1, len(discriminatory_column)):
        if discriminatory_column[i] is not discriminatory_column[i-1]:
            series_sizes.append(i - change_idxs[len(change_idxs)-1])
            change_idxs.append(i)
    series_sizes.append(len(discriminatory_column) - change_idxs[len(change_idxs)-1])
    series_sizes = np.asarray(series_sizes, dtype=int)
    # get the most common sized array and its count
    most_common = Counter(series_sizes).most_common(1)
    standard_size, num_standard_series = most_common[0][0], most_common[0][1]
    # get the indices of the most common array size
    # Note: standard_idxs_table indexes into change_idxs which indexes into closing_column
    standard_idxs = np.where(series_sizes == standard_size)[0]
    if standard_idxs.size != num_standard_series:
        print("Error in your code: Num standard indices as obtained via \
        Counter().most_common does NOT equal number counted from standard_idxs table!")
    return standard_size, num_standard_series, change_idxs, standard_idxs, 

standard_size, num_standard_series, change_idxs, standard_idxs \
    = standard_length(csvfile, name_column)

# function to get data_column and its size parameters
# pass in the desired column name, the desired number of series desired, and the desired length.  If you want all series, pass in 'num_standard_series'. If you want all data points, pass in 'standard_size'
def data_column_and_size(desired_column_name, num_series, num_pts_per_series):
    data_column = csvfile[desired_column_name] 
    return data_column, num_series, num_pts_per_series

closing_column, num_series, num_pts_per_series = data_column_and_size("close", 50, standard_size)

# function to get a 2D arrays of standardized and normalized time series arrays 
def series(data_column, standard_idxs, change_idxs, num_series, num_pts_per_series):
    # create arrays of from the standard-sized closing-price series
    series = []
    for idx in standard_idxs[:num_series]:
        start_idx = change_idxs[idx]
        data = [data_column[start_idx + i] for i in range(0, num_pts_per_series)]
        data = np.asarray(data, dtype=np.double)
        # normalize the array
        data_normalized = stats.zscore(data) 
        # and append the array to 'series'
        series.append(data_normalized)
    series = np.asarray(series, dtype=np.double)
    # Add the absolute value of the min in the series to all values in series so as to make all values in series non-negative. (Needed for the purposes of our d3 code).
    min_val = np.amin(series)
    series += abs(min_val)
    print(np.amax(series))
    return series

series = series(closing_column, standard_idxs, change_idxs, num_series, num_pts_per_series)
# Function to get DTW distance matrix for num_series in series list
# Window size defaults to 10% of the series' length.
# If you don't want a window, set it's size to 1
# Psi (flexibility on the end points) defaults to 1% of the series length + 1.
# If you don't want flexibility, set it's size to 0
# Since the fast version of this function does not support matrix flattening, after 
# getting the matrix, flatten it by creating a 1D array of its upper triangle's values.    
# This is necessary for our purposes because only a flattened matrix can be passed to 
# Scipy's linkage function. 
def distance_matrix(series, window_size=int(num_pts_per_series/10), \
    psi_size=int((num_pts_per_series/100)+1)):
    dm = dtw.distance_matrix_fast(series, window=window_size, psi=psi_size )
    # flatten the matrix 
    dm_flat = []
    for i in range(0, num_series):
        for j in range(i+1, num_series):
            dm_flat.append(dm[i,j])
    dm_flat = np.asarray(dm_flat, dtype=np.double)
    return dm_flat

window_size = 1 
psi_size = 0
dm_flat = distance_matrix(series, window_size, psi_size)   

# Note: If You were using the normal (not fast) version of this function, 
# you could use this code (and comment out the above code):
# Note: In the below code, we used the keyword compact=True' to flatten the matrix 
# and return only the upper triangle (see dtaidistance documentation). This saves 
# memory space. 
  
# dm_flat = dtw.distance_matrix(series[:num_series, :num_pts_per_series], \
#    window=window_size, psi=psi_size, compact=True )


# Function to Get heirarchical linkage table Z, and to plot the dendogram of the linkages. 
# Note thtat the height of each linkage corresponds to distance between its nodes
# See SciPy's cluster.hierarchy.linkage docs for how to read a Z table
# here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
# This blog post explains it more clearly: 
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/ 
# Note: we use 'complete' linkage (vs. single, average, ward, or other) 
# because it increases compactness, and b/c it is compatible with 
# dtaidistance graphings functions
def linkage_and_deondogram(dm_flat, dir_name):
    Z = linkage(dm_flat, 'complete')
    plt.figure(figsize=(10, 4))
    dendrogram(Z, color_threshold=0)
    plt.savefig(dir_name+"/dendogram_scipy_"+str(num_series)+"_"+str(num_pts_per_series)+".png")
    return Z

Z = linkage_and_deondogram(dm_flat, "my_images")

# Function to create dtai dendogram which shows a plot of each series next to 
# its position on the the dendogram. Recommended for small numbers of series (<~25) 
# with few data points (<~100):
def dtai_dendogram(series, dir_name):
    from dtaidistance import clustering
    model = clustering.LinkageTree(dtw.distance_matrix_fast, \
        {'window':window_size, 'psi':psi_size})
    model.fit(series)
    model.plot(filename=dir_name+"/dendogram_dtai_"+str(num_series)+"_"+str(num_pts_per_series)+".png",\
        axes=None, ts_height=.5, \
        bottom_margin=.4, top_margin=.4, ts_left_margin=.2, \
        ts_sample_length=1/num_pts_per_series, \
        tr_label_margin=.1, tr_left_margin=0, ts_label_margin=-.25, \
        show_ts_label=lambda x : "ts-" + str(x), show_tr_label=True, \
        cmap='viridis_r', ts_color=None)

# (Uncomment this code if you want to implement the above funciton)
dtai_dendogram(series, "my_images")

# CLUSTERING: A cluster is a group of leaf nodes in the dendogram that are compact, i.e they all have close distance values relative to each other and far distant values relative to higher links. 
# Algorithm for compactness determination. Author: jweissko
# 1. for each row index of Z see if it represents a link of more than 2 children. 
#    2. If yes, we will call it a parent node. Get the parent node's distance value and see if each of its children nodes are leaves or links.  
#       3. If a child node is a link, then find its distance and divide by its parent's distance to get a fraction between 0 and 1 (the 'proportion')
#       4. Give that child a score of .5 (which is obained by giving it a score of 1 divided by 2 times the distance of the child's generation from the parent)
#    5. Look at the child's children (the parent's grandchildren) to see if they are links or leaves. If a grandchild is a link, then find its distances and divide by its parent (the original child's) distance to get a fraction between 0 and 1. Compare that fraction with it's parent's fraction above. If the grandchild's fraction is at least 10% larger than the child's fraction, then there might be a case for considering the child and grandchild to be compact relative to the first parent. Add the fraction of the grandchild to the score of the child parent, but this time divide it by 4 (which is obtained by taking the distance of the generation of the grandchild (2nd generation from parents) times 2). 
# Recursively repeat this process with the grandchild's own children, and their children, etc., until the offspring fraction is not larger by 10% than the initial child's fraction or until a leaf node is reached. At each stage, add to the original child's score if conditions for doing so are met. 
# Note: Since the limit of the inifinite series {1/2 + 1/4 + 1/8 + 1/16 + ...} approaches 1, we are guaranted that the final score will not exceed 1.
#       6. If the final score of the child > its fraction, then consider it to define a cluster apart from its parent link, and add it's index to the cluster_idxs list.

# Function to get the indexes that defined a cluster
# Param: None
# Output: list of cluster indices
def cluster_idxs(Z):
    cluster_idxs = []
    for i in range(0, len(Z)):
        row = Z[i]
        distance = row[2]
        num_children = row[3]
        if num_children > 2:
            for j in range(0,2):
                child_idx = int(row[j])
                if child_idx >= num_series:
                    child_row = Z[child_idx-num_series]
                    child_dist = child_row[2]
                    proportion = child_dist/distance
                    score = grandchildren_score(Z, child_row, \
                        child_dist, proportion, score=.5, generation=2)
                    if score > proportion:
                        cluster_idxs.append(child_idx)
    return cluster_idxs

# Helper function to get grandchildren's scores added to their parent's score. Called recursively
# Param: row of parent of grandchild, distance of parent of grandchild, distance fraction 
# (i.e. proportion) of original child under question as to whether it should define a 
# cluster, score of original child as accumulated thus far, generation count from original 
# parent index. 
# Output: updated score of original child
def grandchildren_score(Z, row, dist, proportion, score, generation):
    if row[3] > 2:
        for j in range(0,2):
            child_idx = int(row[j])
            if child_idx >= num_series:
                child_row = Z[child_idx-num_series]
                child_dist = child_row[2]
                child_prop = child_dist/dist
                if child_prop > proportion + .1:
                    score += (child_prop / (generation * 2))
                    generation += 1
                    score += grandchildren_score(Z,child_row, \
                        child_dist, proportion, score, generation)
    return score

# Get cluster indexes in Z table 
cluster_idxs = cluster_idxs(Z)

# Recursive helper function to get the mean of the series that compose a cluster. This mean will be used to represent the cluster. 
# Recursive algorithm for mean_series of a cluster, Author jweissko:
# 1. Base case: the cluster has just two series. Find the warping path between the two series, which represents the closest correlation between the series. For each entry of the warping path finding the entries in the series, and get theri values. Average their values, and append that average to the mean series. Do this for all entries in the warping path. Since there are generally more path entries than entries in any one series, and since path entries tend to increse at the points in the path where series don't correlate one to one, the path entries numbers will shift the resultant mean series to the left or right so that its features are positioned at an average time between the series points that represent.
# 2. More complex case: A cluster is made of one or more other other linkages. In this case, the mean series is first calculated recursively for the base case linkages in the cluster, then the means of those base cases are passed up the linkage tree to the higher level linkages to be avearged with the incoming series and means. In other words, in moving up the tree from the leaves, at each link, the mean is calculated between all series and means that are connected to that link. 
# 3. When the average is calcualted, weights are given in proportion to how many series it represents, with each series that it represents increasing the means weight by 1. Thus,iIf a mean represents 3 series, when averaged with another mean that represents 2 series, the mean that represents 3 series will have 3/5 weight in teh resultant mean, and the mean that represents 2 series will have 2/5 weight in teh resultant mean.
# Parameters: series, Z, cluster index (from Z table) that defines a cluster (i.e. all its children nodes are part of the cluster, and no nodes abve it are part of the cluster), dictionary for dynamic reuse of previously calculated mean series
# Output: the mean series, its label (a tuple of tuples of the child ids that compase it), and the mean series's weight (used primarily for the purpose of recursion, but also indicative of how many series are in the cluster that the mean is representing). 
def mean_of_cluster(series, Z, parent_idx, dp):
    if parent_idx >= num_series:
        parent_idx -= num_series
    # if this index has already been calculated, return its results
    if parent_idx in dp:
        return  dp[parent_idx][0],  dp[parent_idx][1],  dp[parent_idx][2]
    child1_idx = int(Z[parent_idx][0])
    child2_idx = int(Z[parent_idx][1])
    # If the child_idx is not a reference to a different index, then it is a reference to a series id, so get that series and give it a weight of 1
    if child1_idx < num_series:
        child1 = series[child1_idx]
        child1_label = child1_idx
        child1_weight = 1
    # If the child_idx is a reference to another index, recursively get the mean_series of that index
    else: 
        child1, child1_label, child1_weight = mean_of_cluster(series, Z, child1_idx, dp)
    if child2_idx < num_series:
        child2 = series[child2_idx]
        child2_label = child2_idx
        child2_weight = 1
    else: 
        child2, child2_label, child2_weight = mean_of_cluster(series, Z, child2_idx, dp)
    # Get the warping path from child1's series to child2's series
    path = dtw.warping_path(child1, child2)
    # the initial mean will be longer than the length of either child series, since it will include every path connection
    mean_long = []
    for pair in path:
        child1_val = child1[pair[0]]
        child2_val = child2[pair[1]]
        # Calculate the mean using the children's weight. Each series that the child includes adds another unit of weight to the child's series
        mid = (child1_val * child1_weight + child2_val * child2_weight) \
            / (child1_weight + child2_weight)
        mean_long.append(mid)
    # transform into np.array to allow for interpolation to be applied to it
    mean_long = np.asarray(mean_long, dtype=np.double)
    # Interpolation creates a function that follows the curve defined be mean_long
    mean_interp = interp.interp1d(np.arange(mean_long.size), mean_long)
    # Sample the function with num_pts_per_series samplings to approximate mean_long in with num_pts_per_series points 
    mean_compress = mean_interp(np.linspace(0, mean_long.size-1, num_pts_per_series))
    # input results to dp dictionary for future reference
    dp[parent_idx] = [mean_compress, (child1_label, child2_label), \
        child1_weight + child2_weight]
    return mean_compress, (child1_label, child2_label), child1_weight + child2_weight

# Function to get the mean_series for each cluster, cluser/mean_series labels, and number of children in each cluster
def mean_series():
    # mean_series for each cluster, label names of mean_series, size of each cluster
    mean_series, mean_labels, mean_num_children = [], [], []
    # dynamic programming dictionary
    dp = {}
    for idx in cluster_idxs:
        mean, label, child_weight = mean_of_cluster(series, Z, idx, dp)
        mean_series.append(mean)
        mean_labels.append(label)
        mean_num_children.append(child_weight)
    mean_series = np.asarray(mean_series)
    return mean_series, mean_labels, mean_num_children

mean_series, mean_labels, mean_num_children = mean_series()

# Function to print the Z table followed by the cluster names and number of series in each cluster to a txt file
def print_Z_and_clusters(dir_name, Z, cluster_idxs, mean_labels, mean_num_children):
    # print the Z table, the cluster indices and chidren quantities to a txt file
    Z_file = open(dir_name+"/Z_table_"+str(num_series)+"_"+str(num_pts_per_series)+".txt","w")
    Z_file.write("Wrtten on %s\n\n" % datetime.datetime.now())
    for i in range(0, len(Z)):
        Z_file.write("%d. %s\n" % (i,Z[i]))
    Z_file.write("\ncluster_idxs: %s\n" % (cluster_idxs))
    Z_file.write("\nmean_lables: \n")
    for i in range(0, len(mean_labels)):
        label = mean_labels[i]
        Z_file.write("%s\n" % (label,))
    Z_file.write("\nmean_num_children:\n")
    Z_file.write("%s\n" % (mean_num_children))
    Z_file.close()

print_Z_and_clusters("my_Z_table_texts", Z, cluster_idxs, mean_labels, mean_num_children)

# function to create a csv of the series and mean series. Each column represents a series. 
# Each row, a date.
def create_csv(csvfile, series, num_series, mean_labels, mean_series):
    # turn the data into a pandas DataFrame for conversion to a csv
    # define the column labels, index, and data
    df_columns = [i for i in range(0, num_series)] + mean_labels
    df_data = np.transpose(np.concatenate((series, mean_series), axis=0))
    df_index = csvfile.date[0:num_pts_per_series]
    # Create dataFrame object and create with it a csv file
    df = pd.DataFrame(data=df_data, index=df_index, columns=df_columns)
    df.to_csv("my_csvfiles/my_csv_"+str(num_series)+"_"+str(num_pts_per_series)+".csv")

create_csv(csvfile, series, num_series, mean_labels, mean_series)

# crate csv with a structure of just three columns: id, date, and price_data, with each date repeaing num_series times before going onto the next date. Do this for the purpose of fitting into Maha's d3 framework
def create_special_format_csv(csvfile, series, num_series, mean_series):
    df_columns = ["id", "date", "closing price"]
    prices = np.concatenate((series, mean_series), axis=0)
    series_count = prices.shape[0]
    prices = np.reshape(prices, (prices.size, 1), order='F')
    ids = [i for i in range(1,num_series+1)] \
        + ["c"+str(i) for i in range(1,mean_series.shape[0]+1)]
    ids *= num_pts_per_series
    ids = np.asarray(ids)
    ids = np.reshape(ids, (ids.size, 1))
    date_reformatter = lambda d: datetime.datetime.strptime(d, '%Y-%m-%d')\
            .strftime('%d-%m-%y')
    dates_slice = list(map(date_reformatter, csvfile.date[:num_pts_per_series]))     
    dates = np.asarray(np.repeat(dates_slice, series_count))
    dates = np.reshape(dates, (dates.size, 1))
    df_data = np.concatenate((ids, dates, prices), axis=1)
    df = pd.DataFrame(data=df_data, columns=df_columns)
    df.to_csv("my_csvfiles/special_format_csv_"+str(num_series)+"_"\
        +str(num_pts_per_series)+".csv", index=False)

create_special_format_csv(csvfile, series, num_series, mean_series)