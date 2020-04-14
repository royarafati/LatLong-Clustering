import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans



#Loading Data :
df= pd.read_csv (r'E:\Arshad\pattern recognition\Poject\Clustering\latLong1.csv')

X=df.loc[:,['Longitude','Latitude']]

# Input and Output data:
X_axis=np.array(df['Longitude'])
Y_axis=np.array(df['Latitude'])

#Elbow Curve :
K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = df[['Latitude']]
X_axis = df[['Longitude']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
# Visualize elbow curve:
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')


#Applying Kmeans:

kmeans = KMeans(n_clusters = 5,)
kmeans.fit(X[X.columns[0:2]]) # Compute k-means clustering.
X['cluster_label'] = kmeans.fit_predict(X[X.columns[0:2]]) #definig labels
centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
labels = kmeans.predict(X[X.columns[0:2]]) # Labels of each point

#visualize
X.plot.scatter(x = 'Longitude', y = 'Latitude', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:,0], centers[:,1],c='black', s=200, alpha=0.5)
plt.show()
