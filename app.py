import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd 
import json
import dash_ag_grid as dag
from dash.exceptions import PreventUpdate


from youtube_scripts import get_video_comments
from utils import convert_url_to_video_id

from nlp_scripts import count_entity_type, apply_ner_functions, count_entities

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = dbc.Container(
    fluid=True,
    style={
        'backgroundImage': 'url(https://example.com/dice-background.jpg)',  # Replace with your dice background URL
        'backgroundSize': 'cover',
        'height': '100vh',
        'textAlign': 'center',
        'paddingTop': '50px',
    },
    children=[
        dcc.Store(id='comments'),
        dcc.Store(id='video-id'),
        dbc.Row(
            dbc.Col(
                html.H1(
                    "YouTube NLP Dashboard",
                    style={'color': 'white', 'fontWeight': 'bold'}
                ),
                width=12
            ),
        ),
        dbc.Row(
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.Input(id="video-url", placeholder="Enter YouTube video URL", type="text"),
                        dbc.Button("Submit", id="submit-url", color="primary", n_clicks=0),
                    ],
                    className="mb-3",
                ),
                width=6,
                className="mx-auto"
            ),
        ),
        dbc.Row(
            dbc.Col(
                dcc.Loading(
                    id="loading-spinner",
                    type="default",
                    children=html.Div(id="loading-output"),
                ),
                width=12
            ),
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dag.AgGrid(
                            id='comments-grid',
                            style={"height": "500px", "width": "100%"},
                            columnDefs=[
                                {"headerName": "Comment", "field": "textOriginal", "sortable": True, "filter": True, "tooltipField": "textOriginal"},
                                {"headerName": "Likes", "field": "likeCount", "sortable": True, "filter": True},
                                {"headerName": "Published At", "field": "publishedAt", "sortable": True, "filter": True},
                                {"headerName": "Sentiment", "field": "sentiment", "sortable": True, "filter": True}
                            ],
                            dashGridOptions={
                                "pagination": True,
                                "paginationPageSize": 10,
                                "enableBrowserTooltips": True  # Enable tooltips
                            }
                        ),
                    ],
                    width=6,
                    style={'padding': '10px'}
                ),
                dbc.Col(
                    html.Div(id='video-container'),
                    width=6,
                    style={'padding': '10px'}
                ),
            ]
        ),

           dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id='sentiment-over-time',
                        figure={
                            'data': [{'x': [], 'y': [], 'type': 'bar', 'name': 'Placeholder'}],
                            'layout': {'title': 'Placeholder Chart 1'}
                        }
                    ),
                    width=6,
                    style={'padding': '10px'}
                ),
                dbc.Col(
                    dcc.Graph(
                        id='sentiment-histogram',
                        figure={
                            'data': [{'x': [], 'y': [], 'type': 'bar', 'name': 'Placeholder'}],
                            'layout': {'title': 'Placeholder Chart 2'}
                        }
                    ),
                    width=6,
                    style={'padding': '10px'}
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id='name-entity-counts',
                        figure={
                            'data': [{'x': [], 'y': [], 'type': 'bar', 'name': 'Placeholder'}],
                            'layout': {'title': 'Placeholder Chart 1'}
                        }
                    ),
                    width=6,
                    style={'padding': '10px'}
                ),
                dbc.Col(
                    dcc.Graph(
                        id='top-30-named-entities',
                        figure={
                            'data': [{'x': [], 'y': [], 'type': 'bar', 'name': 'Placeholder'}],
                            'layout': {'title': 'Placeholder Chart 2'}
                        }
                    ),
                    width=6,
                    style={'padding': '10px'}
                ),
            ]
        )
    ]
)

# Callback to handle URL submission and store comments
@app.callback(
    [Output('comments', 'data'),
     Output('loading-output', 'children'),
     Output('submit-url', 'disabled'),
     Output('video-id', 'data')],
    [Input('submit-url', 'n_clicks')],
    [State('video-url', 'value')]
)
def update_comments(n_clicks, url):
    if n_clicks > 0 and url:
        video_id = convert_url_to_video_id(url)
        comments = get_video_comments(video_id)
        comments = apply_ner_functions(comments)
        comments['sentiment'] = comments['textDisplay'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        return comments.to_dict(orient='records'), "", False, video_id
    return None, "", False, None

# Callback to update video iframe
@app.callback(
    Output('video-container', 'children'),
    [Input('video-id', 'data')]
)
def update_video_container(video_id):
    if video_id:
        iframe = html.Iframe(
            src=f"https://www.youtube.com/embed/{video_id}",
            style={"width": "100%", "height": "500px", "border": "none"}
        )
        return iframe
    return ""

# Callback to update the AG-Grid with comments
@app.callback(
    Output('comments-grid', 'rowData'),
    [Input('comments', 'data')]
)
def update_comments_grid(data):
    if data:
        return data
    return []


@app.callback(
        Output('sentiment-over-time', 'figure'),
        Output('sentiment-histogram', 'figure'),
        Input('comments', 
              'data')
)
def plot_sentiment_charts(data):

    if not data:
        raise PreventUpdate
    
    data = pd.DataFrame(data)
    data.index = pd.to_datetime(data.publishedAt)

    data['sentiment_through_time'] = data['sentiment'].rolling(20).mean()
    # make the sentiment over time plot
    fig_1 = go.Figure()
    fig_1.add_trace(go.Scatter(x=data.index, y=data['sentiment_through_time'], mode='lines+markers', name='Sentiment'))
    fig_1.update_layout(
        title='Sentiment Over Time',
        xaxis_title='Time',
        yaxis_title='Sentiment',
        template='plotly'
    )
     # Create the sentiment histogram
    fig_2 = go.Figure()
    fig_2.add_trace(go.Histogram(x=data['sentiment'], nbinsx=20))
    fig_2.update_layout(
        title='Sentiment Distribution',
        xaxis_title='Sentiment',
        yaxis_title='Frequency',
        template='plotly'
    )

    return fig_1, fig_2


# Callback to update the graphs
@app.callback(
    [Output('name-entity-counts', 'figure'),
     Output('top-30-named-entities', 'figure')],
    [Input('comments', 'data')]
)
def update_comment_figures_row_2(data):
    if data:
        data = pd.DataFrame(data)

        # Top entities
        labels, counts = count_entities(df=data)

        # Create the Plotly bar chart
        fig_topn = go.Figure()

        fig_topn.add_trace(go.Bar(
            y=labels,
            x=counts,
            orientation='h',
            marker=dict(color='skyblue')
        ))

        # Update layout
        fig_topn.update_layout(
            title=f'Top {len(counts)} Named Entities',
            xaxis_title='Count',
            yaxis_title='Named Entity',
            yaxis=dict(autorange='reversed'),  # Invert y-axis to display top entities at the top
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Entity type counts
        labels, counts = count_entity_type(df=data)

        fig_entity_type = go.Figure()

        fig_entity_type.add_trace(go.Bar(
            y=labels,
            x=counts,
            orientation='h',
            marker=dict(color='lightgreen')
        ))

        # Update layout
        fig_entity_type.update_layout(
            title=f'Entity Type Counts',
            xaxis_title='Count',
            yaxis_title='Entity Type',
            yaxis=dict(autorange='reversed'),  # Invert y-axis to display top entities at the top
            margin=dict(l=40, r=40, t=40, b=40)
        )

        return fig_entity_type, fig_topn
    return {}, {}

if __name__ == '__main__':
    app.run_server(debug=True, port=8060)