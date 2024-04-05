import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_auth
from dotenv import load_dotenv
import os


# Load the environment variables from .env file
load_dotenv()

# Now you can use os.environ to get environment variables
SECRET_KEY = os.environ.get('DASH_SECRET_KEY', 'default_secret_key')

# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = {
    'Enrico': 'topsecret',
    'Luca':'topsecret',
    'Pascal':'topsecret'
}


app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB])
server = app.server
server.secret_key = SECRET_KEY
auth = dash_auth.BasicAuth(
    app,
    {
        'Enrico': 'topsecret',
        'Luca':'topsecret',
        'Pascal':'topsecret'
    }
)

# Define the sidebar navigation
sidebar = dbc.Nav(
    [
        dbc.NavLink(
            [
                html.Div(page["name"], className="ms-2"),
            ],
            href=page["path"],
            active="exact",
        )
        for page in dash.page_registry.values()
    ],
    vertical=True,
    pills=True,
    className="bg-light",
)

# Define the header
header = dbc.Row(
    dbc.Col(
        html.H2("Live Engagement & Insight Dashboards For Study With Me",
                className='text-center py-3 my-2',
                style={'color': '#4c72b0', 'fontSize': 30}),
        width=12
    )
)

# Combine sidebar and page content in a layout
app.layout = dbc.Container(
    [
        header,
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(sidebar, xs=12, sm=4, md=2, lg=2, xl=2, xxl=2),
                dbc.Col(dash.page_container, xs=12, sm=8, md=10, lg=10, xl=10, xxl=10)
            ]
        )
    ],
    fluid=True,
    className='dbc'
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8071)
