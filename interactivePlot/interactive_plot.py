import pandas as pd
import plotly.express as px

class InteractivePlot:
    def interactive_plot(self, embeddings, labels, filenames):
        df = pd.DataFrame({
            'filename': filenames,
            'cluster': labels,
            'x': [e[0] for e in embeddings],
            'y': [e[1] for e in embeddings],
        })

        fig = px.scatter(df, x='x', y='y', color=df['cluster'].astype(str),
                        hover_data=['filename'], title="Interaktive Cluster-Visualisierung")
        fig.show()
