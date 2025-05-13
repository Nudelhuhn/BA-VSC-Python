import pandas as pd
import plotly.express as px

class InteractivePlot:
    def interactive_plot(self, embeddings, labels, filenames, parent_dirs):
        df = pd.DataFrame({
            'filename': filenames,
            'parent_dir': parent_dirs,
            'cluster': labels,
            'x': [e[0] for e in embeddings],
            'y': [e[1] for e in embeddings],
        })

        # Zeige sowohl den Dateinamen als auch den übergeordneten Ordnernamen im Hover-Text an
        fig = px.scatter(df, x='x', y='y', color=df['cluster'].astype(str),
                         hover_data=['filename', 'parent_dir'], title="Interaktive Cluster-Visualisierung")
        fig.show()
