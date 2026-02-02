import plotly.graph_objects as go
import plotly.offline


def plot_tokens(sentences_data, title, dims=[0,1,2]):

    data=[
        go.Scatter3d(
            x=[s[dims[0]] for s in sentence_data['embeddings']],
            y=[s[dims[1]] for s in sentence_data['embeddings']],
            z=[s[dims[2]] for s in sentence_data['embeddings']],
            mode='markers+text',
            text=sentence_data['labels'],
            textposition="top center",
            marker=dict(size=6, color=sentence_data["color"])
        )for sentence_data in sentences_data
    ]
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='Dim {}'.format(dims[0]),
            yaxis_title='Dim {}'.format(dims[1]),
            zaxis_title='Dim {}'.format(dims[2]),
        ),
        title= title,
        width=1000,
        height=1000,
    )

    fig= go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig)