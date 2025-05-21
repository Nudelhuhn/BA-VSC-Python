import matplotlib.pyplot as plt # needed for plotting/displaying and visualization
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

class ClusterPlotter:
    @staticmethod   # static method since no object of this class is needed, because there is no need to save internal states
    def plot(reduced_embeddings, labels):
        plt.figure(figsize=(10, 7)) # create visualization window

        if reduced_embeddings.shape[1] == 3:    # three or two dimensions?
            axis = plt.subplot(111, projection='3d')  # create three axis plot object
            scatter = axis.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], # x-, y-, z-coordinates
                                   c=labels,    # the color of each point is determined by the respective cluster label
                                   cmap='tab10')    # color palette for up to 10 different categories
            plt.title("3D Cluster Visualization")
            cbar = plt.colorbar(scatter, ax=axis)  # creates a colorbar to show which cluster-label corresponds to which color
            cbar.set_label('Cluster Label') # give a title to the colorbar
        else:
            scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='tab10')
            plt.title("2D Cluster Visualization")
            plt.colorbar(scatter)
        
        plt.show()  # show the finished plot
