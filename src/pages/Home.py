# pages/home.py
import dash
from dash import html

dash.register_page(__name__, path='/')

layout = html.Div([
    html.H1("Welcome to the YouTube Live Stream Insight Dashboard"),
    html.P("Discover insights and engage better with your live streaming audience."),
    # Add more engaging content and visuals as needed
])
