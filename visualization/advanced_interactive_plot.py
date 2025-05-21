import pandas as pd # library for tables and data analysis
import plotly.express as px # simple interface for interactive visualization

class AdvancedInteractivePlot:
    @staticmethod
    def ad_int_plot(embeddings, labels, filenames, parent_dirs):
        df = pd.DataFrame({
            'filename': filenames,
            'parent_dir': parent_dirs,
            'cluster': labels,
            'x': [e[0] for e in embeddings],    # first dimension
            'y': [e[1] for e in embeddings],    # second dimension
            'z': [e[2] for e in embeddings],    # if a third dimension should be shown, change the n_components values in config.yaml to value 3
        })

        # 2D scatter, creates scattered diagram of points
        # fig = px.scatter(df, x='x', y='y',  # dimensions
        #                  color=df['cluster'].astype(str),   # cluster labels
        #                  hover_data=['filename', 'parent_dir'], # data which is shown if hovered over the point
        #                  title="Interaktive Cluster-Visualisierung")
        
        # 3D scatter
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color=df['cluster'].astype(str),
            hover_data=['filename', 'parent_dir'],
            title="Interaktive 3D-Cluster-Visualisierung"
        )
        
        fig.show()  # show the finished plot
