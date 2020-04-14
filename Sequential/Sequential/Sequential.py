import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans



#Loading Data :
df= pd.read_csv (r'E:\Arshad\pattern recognition\Poject\Clustering\latLong1.csv')

X=df.loc[:,['Longitude','Latitude']]

# Hierachical Clustering:
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X, 'single')

labelList = range(1,50)

plt.figure(figsize=(10, 7))
#dendrogram(linked,truncate_mode='lastp',p=25,distance_sort='descending',show_leaf_counts=True)



# set cut-off to 50
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


max_d = 4.95  # max_d as in max_distance

fancy_dendrogram(
    linked,
    truncate_mode='lastp',
    p=25,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()