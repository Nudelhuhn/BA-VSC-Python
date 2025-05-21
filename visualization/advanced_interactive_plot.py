import pandas as pd # library for tables and data analysis
import plotly.express as px # simple interface for interactive visualization

class AdvancedInteractivePlot:
    @staticmethod
    def adv_int_plot(embeddings, labels, filenames, parent_dirs):
        df = pd.DataFrame({
            'filename': filenames,
            'parent_dir': parent_dirs,
            'cluster': labels,
            'x': [e[0] for e in embeddings],    # first dimension
            'y': [e[1] for e in embeddings],    # second dimension
            # 'z': [e[2] for e in embeddings],    # if a third dimension should be shown, change "n_components" in config.yaml to 3
        })

        # decomment for a 2D diagram
        fig = px.scatter(df, x='x', y='y',  # dimensions
                         color=df['cluster'].astype(str),   # cluster labels
                         hover_data=['filename', 'parent_dir'], # data which is shown if hovered over the point
                         title="Interaktive Cluster-Visualisierung")
        
        # # decomment for a 3D diagram
        # fig = px.scatter_3d(
        #     df, x='x', y='y', z='z',
        #     color=df['cluster'].astype(str),
        #     hover_data=['filename', 'parent_dir'],
        #     title="Interaktive 3D-Cluster-Visualisierung")
        
        fig.show()  # show the finished plot
