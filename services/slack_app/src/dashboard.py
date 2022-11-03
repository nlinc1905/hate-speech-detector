from dash import Dash, html, dcc
from dash.dash_table import DataTable
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import sqlite3
import dash_cytoscape as cyto


DB_NAME = "hate_speech_tracking.db"


def convert_edges_df_to_cytoscape_data(df):
    """
    Output format:
        [
            {"data": {"id": "username1", "label", "username1"}},
            {"data": {"id": "username2", "label", "username2"}},
            {"data": {"source": "username1", "target", "username2"}},
        ]
    """
    nodes = set()
    edges = []
    for row in df.iterrows():
        nodes.add(row[1]['source'])
        nodes.add(row[1]['target'])
        edges.append({"data": {"source": row[1]['source'], "target": row[1]['target']}})
    nodes = [{"data": {"id": n, "label": n}} for n in nodes]
    return nodes + edges


biggest_haters_query = (
    "SELECT user, COUNT(*) AS incidents FROM hate_speech GROUP BY user ORDER BY incidents DESC LIMIT 10;"
)
incidents_query = (
    "SELECT timestamp, user, text FROM hate_speech ORDER BY timestamp ASC;"
)
edges_query = (
    "SELECT source, target, COUNT(*) AS nbr_messages FROM edges GROUP BY source, target;"
)

conn = sqlite3.connect(DB_NAME)

biggest_haters_df = pd.read_sql_query(biggest_haters_query, conn)
incidents_df = pd.read_sql_query(incidents_query, conn)
edges_df = pd.read_sql_query(edges_query, conn)
network_data = convert_edges_df_to_cytoscape_data(df=edges_df)


timeline_df = incidents_df.groupby('timestamp')['user'].count().reset_index()
timeline_df.rename(columns={"user": "nbr_incidents"}, inplace=True)
timeline_bar_chart = px.bar(timeline_df, x='timestamp', y='nbr_incidents')


app = Dash(__name__)
server = app.server


app.layout = html.Div(
    children=[
        html.Div(
            id="first_row",
            children=[
                html.Div(
                    id="biggest_haters_container",
                    children=[
                        html.H3("Biggest Haters"),
                        DataTable(id="haters", data=biggest_haters_df.to_dict(orient='records')),
                    ],
                    style={
                        "float": "left",
                    }
                ),
                html.Div(
                    id="network_of_hate_container",
                    children=[
                        html.H3("Network of Hate"),
                        cyto.Cytoscape(
                            id="network",
                            elements=network_data,
                            layout={'name': 'breadthfirst'},
                            style={'width': '600px', 'height': '200px'}
                        )
                    ],
                    style={
                        "float": "right",
                    }
                ),
            ],
            style={
                "padding-left": "2em",
                "padding-right": "2em",
            }
        ),
        html.Div(
            id="second_row",
            children=[
                html.Div(
                    id="timeline_of_hate_container",
                    children=[
                        html.H3("Timeline of Hate"),
                        dcc.Graph(id="timeline", figure=timeline_bar_chart, style={"height": "15vw"}),
                        DataTable(id="incidents", data=incidents_df.to_dict(orient='records')),
                    ]
                ),
            ],
            style={
                "clear": "both",
                "padding-left": "2em",
                "padding-right": "2em",
                "padding-bottom": "2em",
            }
        ),

        dcc.Interval(id="data-update-interval", interval=3*1000, n_intervals=0),

    ],
    style={
        "border": "1px solid gray",
        "border-radius": "0.50%",
    }
)


@app.callback(
    [
        Output('haters', 'data'),
        Output('network', 'elements'),
        Output('timeline', 'figure'),
        Output('incidents', 'data'),
    ],
    Input('data-update-interval', 'n_intervals')
)
def update_dashboard(n):
    """Callback to automatically update the data in the dashboard every n seconds."""
    biggest_haters_df = pd.read_sql_query(biggest_haters_query, conn)
    incidents_df = pd.read_sql_query(incidents_query, conn)
    edges_df = pd.read_sql_query(edges_query, conn)
    network_data = convert_edges_df_to_cytoscape_data(df=edges_df)

    timeline_df = incidents_df.groupby('timestamp')['user'].count().reset_index()
    timeline_df.rename(columns={"user": "nbr_incidents"}, inplace=True)
    timeline_bar_chart = px.bar(timeline_df, x='timestamp', y='nbr_incidents')
    return (
        biggest_haters_df.to_dict(orient='records'),
        network_data,
        timeline_bar_chart,
        incidents_df.to_dict(orient='records')
    )


if __name__ == '__main__':
    # app.run_server(debug=False)
    app.run(host="0.0.0.0", port=8050, debug=False)
