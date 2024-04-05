from dash import dcc, html, Input, Output, dash_table, callback
import plotly.express as px
import pandas as pd
import pytz
import plotly.graph_objs as go
from datetime import datetime
import dash


# Register your page with the pathname of /emotion
dash.register_page(__name__, path='/emotion')



date = "20240317"
csv_file_path = f"csv_input\\emotional_{date}.csv"
data = pd.read_csv(csv_file_path)

data['Hour'] = pd.to_datetime(data['Hour'])
data_agg = data.groupby([data['Hour'].dt.floor('h'), 'PredictedEmotion']).size().reset_index(name='Count')

data_agg['Hour'] = data_agg['Hour'].dt.strftime('%H:00')

data_agg.columns = ['Hour', 'PredictedEmotion', 'Count']




# Define app layout
layout = html.Div([
    html.Div([
        html.H1("Emotional Landscape: Overview Dashboard"),
        html.P(f"Current Date and Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               style={'textAlign': 'right', 'color': 'grey', 'fontSize': '16px', 'display': 'inline-block'})
        ,dcc.Graph(
            id='emotion-count',
            figure=px.bar(
                data_agg.groupby('PredictedEmotion')['Count'].sum().reset_index(),
                x='PredictedEmotion',
                y='Count',
                color='PredictedEmotion',
                title="Total Emotion Count"

            )



        )
    ]),

    html.Div([
        html.H1("Emotion Evolution Through Time"),
        html.P("Times displayed are in UTC format", style={'textAlign': 'right','color': 'grey', 'fontSize': '12px','display': 'inline-block'   }),
        html.P(" "),
        html.P(" "),
        dcc.Dropdown(
            id='emotion-dropdown-time',
            options=[{'label': emotion, 'value': emotion} for emotion in data_agg['PredictedEmotion'].unique()],
            value=data_agg['PredictedEmotion'].unique()[0],
            clearable=False,
            style={'width': '50%', 'display': 'inline-block', 'fontSize': '1.2em'}   # Adjust the width as needed

        )

        ,
        dcc.Graph(id='emotion-count-time'),
    ]),
    html.Div([
        html.H1("Emotion Evolution Across All Emotions"),
        html.P(" "),
        html.P(" "),
        html.P("Times displayed are in UTC format", style={'textAlign': 'right','color': 'grey', 'fontSize': '12px' }),

        dcc.Graph(id='emotion-count-across')
    ])
    ,
    html.Div([
        html.H1("Voices Behind Emotions: Audience Insights"),
        dcc.Dropdown(
            id='emotion-dropdown',
            options=[{'label': emotion, 'value': emotion} for emotion in data['PredictedEmotion'].unique()],
            value=data['PredictedEmotion'].unique()[0],
            clearable=False,
            style={'width': '50%', 'display': 'inline-block', 'fontSize': '1.2em'}
        ),
        html.Div(id='sample-messages')
    ])
])





@callback(
    Output('emotion-count-time', 'figure'),
    [Input('emotion-dropdown-time', 'value')]
)
def update_emotion_time(selected_emotion):
    filtered_data = data_agg[data_agg['PredictedEmotion'] == selected_emotion]

    # Check if filtered data has at least 2 data points
    if len(filtered_data) < 2:
        # Display a message for insufficient data
        return {
            'data': [],
            'layout': {
                'title': 'Insufficient data for selected time window and emotion.'
            }
        }

    # Proceed with creating the graph if data points are sufficient
    fig = px.line(
        filtered_data,
        x='Hour',
        y='Count',
        color='PredictedEmotion',
        title=f"Emotion Count Over Time for {selected_emotion}"
    )
    fig.update_traces(mode='lines+markers')
    fig.update_layout(yaxis=dict(tickmode='linear', tick0=0, dtick=10))

    return fig

# Callback for Emotion Evolution Across All Emotions
@callback(
    Output('emotion-count-across', 'figure'),
    Input('emotion-dropdown-time', 'value')
)
def update_emotion_across(selected_emotion):
    fig = px.line(
        data_agg,
        x='Hour',
        y='Count',
        color='PredictedEmotion',
        title='Emotion Count Over Time Across Emotions'

    )
    fig.update_traces(mode='lines+markers', line=dict(width=2))
    # Update the layout to have a clearer distinction between lines
    fig.update_layout(
        yaxis=dict(tickmode='linear', tick0=0, dtick=10),
        xaxis=dict(tickangle=-45),  # Tilt the x-axis labels for better visibility
        legend=dict(
            title='Emotions',  # Legend title
            orientation='h',   # Horizontal orientation
            yanchor="bottom",
            y=1.02,            # Position the legend above the plot
            xanchor="right",
            x=1
        )
    )
    return fig


# Callback for Sample Processed Messages
@callback(
    Output('sample-messages', 'children'),  # Targeting the 'sample-messages' container for both messages and the table
    [Input('emotion-dropdown', 'value')]
)
def update_sample_messages(selected_emotion):
    filtered_data = data[data['PredictedEmotion'] == selected_emotion]

    # Directly use messages from filtered_data for display
    messages = filtered_data['MessageContent'].tolist()

    # Initialize lists for usernames and user roles
    usernames, userroles = [], []

    # Apply sample size limitation for 'neutral' emotion
    limit_sample = selected_emotion == "neutral"
    sample_size = min(5, len(filtered_data)) if limit_sample else len(filtered_data)

    for message in filtered_data['MessageContent'].tolist()[:sample_size]:  # Apply sample size limitation here
        parts = message.split(':', 1)
        username, role = "Unknown", "Unknown"
        if len(parts) > 1:
            username = parts[0]
            if "(SPONSOR)" in username:
                role = "SPONSOR"
                username = username.replace("(SPONSOR)", "").strip()
            elif "(MODERATOR, SPONSOR)" in username:
                role = "MODERATOR, SPONSOR"
                username = username.replace("(MODERATOR, SPONSOR)", "").strip()
            elif "(Verified, Owner)" in username:
                role = "Verified, Owner"
                username = username.replace("(Verified, Owner)", "").strip()
            else:
                role = "Normal Audience"
        else:
            # If no ':' found, treat the entire message as content with "Unknown" role
            role = "Normal Audience"

        usernames.append(username)
        userroles.append(role)

        # Create DataFrame for DataTable (distincttttt )
        user_info_df = pd.DataFrame({
            'UserName': usernames,
            'UserRole': userroles
        }).drop_duplicates().reset_index(drop=True)

    # Messages component without alteration
    messages_component = html.Ul([html.Li(msg) for msg in messages[:sample_size]])

    # Enhanced DataTable with filtering and sorting
    table_component = dash_table.DataTable(
        columns=[
            {'name': 'UserName', 'id': 'UserName'},
            {'name': 'UserRole', 'id': 'UserRole'}
        ],
        data=user_info_df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
            'overflow': 'hidden', 'textOverflow': 'ellipsis'
        },
        filter_action="native",  # Enable filtering
        sort_action="native",  # Enable sorting
        style_header_conditional=[  # Applying color to column headers
            {
                'if': {'column_id': 'UserName'},
                'backgroundColor': '#FFDDC1',  # Example color
                'color': 'black'
            },
            {
                'if': {'column_id': 'UserRole'},
                'backgroundColor': '#C1FFD7',  # Example color
                'color': 'black'
            }
        ]
    )

    # Combine both components into one container to return
    combined_output = html.Div([
        messages_component,
        html.Br(),  # Spacer between components
        table_component
    ])

    return combined_output



