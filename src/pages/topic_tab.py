from dash import dcc, html, Input, Output, dash_table, callback
import plotly.express as px
import pandas as pd
import pytz
import plotly.graph_objs as go
from datetime import datetime



import ast
import dash
from dash import dcc, html


from datetime import datetime
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import json
import numpy as np
from dash.dependencies import Input, Output ,State
import random
import plotly.graph_objs as go
import dash_cytoscape as cyto
from dash import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os
# Register your page with the pathname of /emotion
dash.register_page(__name__, path='/topic')
def load_and_structure_data(filepath):
    df = pd.read_csv(filepath)
    df['Topic Identifier'] = df['Topic Identifier'].astype(int)
    df['Topic Number'] = df['Topic Number'].astype(int)
    df['Topic Size'] = df['Topic Size'].astype(int)
    df['Representative Doc'] = df['Representative Doc'].astype(str)
    df['Original MessageContent'] = df['Original MessageContent'].astype(str)
    df['PublishedAt'] = pd.to_datetime(df['PublishedAt'])
    df['Label'] = df['Label'].astype(str)
    df['Average Probability'] = df['Average Probability'].astype(float)  # Conditional handling

    # Load and convert "Mentions of Docs Per Hour" from string representation of a dictionary to a dictionary
    if 'Mentions of Docs Per Hour' in df.columns:
        df['Mentions of Docs Per Hour'] = df['Mentions of Docs Per Hour'].apply(ast.literal_eval)
    return df


def extract_user_info(df):
    usernames = []
    user_roles = []

    # Patterns for roles that may appear in the usernames
    role_patterns = {
        "MODERATOR SPONSOR": ["MODERATOR SPONSOR"],
        "Verified Owner": ["Verified Owner"],
        "SPONSOR": ["SPONSOR"],
    }

    for content in df['Original MessageContent']:
        # Initialize default role and message
        user_role = "Normal Audience"
        message = ""

        # Split the content by the first colon to separate username from message
        parts = content.split(':', 1)
        if len(parts) > 1:
            username, message = parts[0], parts[1]
        else:
            username = parts[0]  # No colon found, entire content is considered username

        # Check and replace roles in the username
        for role, patterns in role_patterns.items():
            for pattern in patterns:
                if pattern in username:
                    user_role = role.replace("(", "").replace(")", "")  # Set role, remove parentheses for uniformity
                    username = username.replace(pattern, "").strip()  # Clean the username by removing the role
                    break  # Break after the first match

        usernames.append(username)
        user_roles.append(user_role)

    df['UserName'] = usernames
    df['UserRole'] = user_roles
    return df

def generate_color():
    # Generates a random color
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# date = get_Run_Dt()
# df_original = get_processed_df()
#
# flag = load_flag_value()
#
# if flag == '13' or flag is None:
#     ## Run Initial mode
#     topic_model_initials = generate_topic_barchart_initial(df_original)
#     df_initialize_topic_data = create_topic_info_dataframe(topic_model_initials, df_original)
#     df_topic_data = df_initialize_topic_data
#     save_flag_value('0')
#
# elif flag == '0' or flag == '1':
#     ## Run update model
#     topic_model_updated_per_run, filtered_data_per_run = generate_topic_barchart_updated(df_original)
#     df_Updated_dataInfo_per_run = update_topic_info_dataframe(topic_model_updated_per_run, filtered_data_per_run)
#     df_topic_data = df_Updated_dataInfo_per_run
#     save_flag_value('1')
# else:
#     print("something wrong with the flag")
#     pass
#
#
# print(flag)
date = "20240317"

#csv_file_path = f"csv_input\\topic_modeling_{date}.csv"
csv_file_path = 'topic_m_20240317.csv'
data = pd.read_csv(csv_file_path)
#df_topic_data = load_and_structure_data(f"csv_input\\Topic_Modeling_Analysis_{date}.csv")
#df_topic_data = pd.read_csv(csv_file_path)
load_and_structure_data
df_topic_data = load_and_structure_data(csv_file_path)
df_topic_data['Topic Color'] = df_topic_data['Topic Identifier'].apply(lambda _: generate_color())
df_topic_data['PublishedAt'] = pd.to_datetime(df_topic_data['PublishedAt'])
df_topic_data['Hour'] = df_topic_data['PublishedAt'].dt.strftime('%H:00')



##############################Function###########


# Call the function to extract user information

df_topic_data = extract_user_info(df_topic_data)







def generate_bar_chart(df, max_topic_id):
    # Filter based on the slider value
    df = df[df['Topic Identifier'] <= max_topic_id]

    # Drop duplicate labels within the filtered DataFrame
    df_unique_labels = df.drop_duplicates(subset=['Label'])

    # Generate the bar chart
    fig = go.Figure()
    for _, row in df_unique_labels.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Topic Size']],
            y=[f"{row['Topic Identifier']}: {row['Label']}"],
            name=f"Topic {row['Topic Identifier']}",  # This will set the legend name as "Topic <identifier>"
            orientation='h',
            marker_color=row['Topic Color']
        ))
    fig.update_layout(
        title="Number of Documents per Topic",
        xaxis_title="Number of Documents",
        yaxis_title="Topics Labels",
        yaxis=dict(autorange="reversed")
    )
    return fig


@callback(
    Output('network-graph-container', 'children'),
    [Input('topic-identifier-slider', 'value'),
     Input('topic-label-dropdown_graph', 'value'),
     Input('username-dropdown', 'value'),
     Input('refresh-button', 'n_clicks')]
)
def update_network_graph(slider_value, selected_labels, selected_usernames, n_clicks):
    # Filter the DataFrame based on inputs
    filtered_df = df_topic_data.copy()
    if selected_labels:
        filtered_df = filtered_df[filtered_df['Label'].isin(selected_labels)]
    if selected_usernames:
        filtered_df = filtered_df[filtered_df['UserName'].isin(selected_usernames)]
    if slider_value is not None:
        # Keep topics with an identifier less than or equal to the slider value
        filtered_df = filtered_df[filtered_df['Topic Identifier'] <= slider_value]

    # Generate and return the updated network graph
    return generate_network_graph(filtered_df)




def generate_network_graph(df):
    nodes = []
    edges = []

    # Define colors for Topics and Users
    topic_color = "#636EFA"
    user_color = "#EF553B"

    # Ensure the DataFrame is not empty
    if not df.empty:
        # Create nodes for topics
        for _, row in df.iterrows():
            nodes.append({
                "data": {"id": f"Topic {row['Topic Identifier']}", "label": row['Label']},
                "classes": "topic"  # Class for styling
            })

        # Create nodes for users and edges between users and topics
        # Ensure we're only adding unique usernames as nodes to avoid duplicates
        unique_usernames = df['UserName'].unique()
        for username in unique_usernames:
            nodes.append({
                "data": {"id": username, "label": username},
                "classes": "user"  # Class for styling
            })

        # Create edges based on the filtered DataFrame
        for _, row in df.iterrows():
            edges.append({
                "data": {"source": row['UserName'], "target": f"Topic {row['Topic Identifier']}"},
                "classes": "interaction"  # Class for styling
            })

    elements = nodes + edges

    return cyto.Cytoscape(
        id='network-graph',
        elements=elements,
        style={'width': '100%', 'height': '400px'},
        layout={'name': 'breadthfirst'},
        stylesheet=[
            {'selector': 'node.topic', 'style': {'background-color': topic_color, 'label': 'data(label)'}},
            {'selector': 'node.user', 'style': {'background-color': user_color, 'label': 'data(label)'}},
            {'selector': 'edge.interaction', 'style': {'line-color': user_color}},
        ]
    )





@callback(
    Output('topics-bar-chart', 'figure'),
    [Input('topic-slider', 'value')]
)
def update_output(selected_topic_id):
    # Generate the bar chart with unique labels up to the selected topic ID
    return generate_bar_chart(df_topic_data, selected_topic_id)


from collections import defaultdict


@callback(
    Output('time-series-chart', 'figure'),
    [Input('time-series-chart', 'id')]  # Input is just a dummy to trigger the callback
)

def update_time_series(_):
    # Assuming df_topic_data is available here and is up-to-date
    # You may need to load it from a file or pass it as an input to the callback if necessary

    # Extract mentions per hour across topics
    mentions_data = defaultdict(lambda: defaultdict(int))
    for _, row in df_topic_data.iterrows():
        topic_label = row['Label']
        mentions_dict = row['Mentions of Docs Per Hour']  # This should already be a dictionary
        for hour, count in mentions_dict.items():
            mentions_data[topic_label][hour] += count

    # Create traces for the line chart for each topic label
    data = []
    all_labels = sorted(df_topic_data['Label'].unique())
    default_labels = all_labels[:2]  # The first two labels will be visible by default

    for topic_label in all_labels:
        hours_mentions = mentions_data[topic_label]
        sorted_hours = sorted(hours_mentions.keys())
        is_visible = True if topic_label in default_labels else 'legendonly'
        trace = go.Scatter(
            x=sorted_hours,  # Sorted hours for the x-axis
            y=[hours_mentions[hour] for hour in sorted_hours],  # Corresponding mention counts for the y-axis
            mode='lines+markers',
            name=topic_label,
            visible=is_visible
        )
        data.append(trace)

    # Define the layout for the plot
    layout = go.Layout(
        title='Topic Engagement Over Time',
        xaxis=dict(title='Hour of the Day'),
        yaxis=dict(title='Number of Mentions'),
        hovermode='closest',
        legend=dict(
            traceorder='normal',
            itemsizing='constant'
        )
    )

    # Return the figure
    return {'data': data, 'layout': layout}


@callback(
    Output('details-table', 'data'),
    [Input('topic-label-dropdown', 'value')]
)
def update_table(selected_labels):
    # Filter df_topic_data based on the selected labels
    if selected_labels:
        filtered_data = df_topic_data[df_topic_data['Label'].isin(selected_labels)]
    else:
        filtered_data = df_topic_data

    # Convert the filtered DataFrame to a list of dictionaries for DataTable
    table_data = filtered_data[['Original MessageContent', 'UserName', 'UserRole']].to_dict('records')
    return table_data


layout = html.Div([
    html.Div([
        html.H1("Engagement in Numbers: Streaming Topics Analyzed"),
        html.P(f"Current Date and Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               style={'textAlign': 'right', 'color': 'grey', 'fontSize': '16px', 'display': 'inline-block'}),
        dcc.Slider(
            id='topic-slider',
            min=1,
            max=df_topic_data['Topic Identifier'].max(),
            value=df_topic_data['Topic Identifier'].max(),
            marks={str(i): {'label': str(i), 'style': {'color': '#77b0b1'}}
                   for i in df_topic_data['Topic Identifier'].unique()},
            step=1,
        ),
        dcc.Graph(id='topics-bar-chart')
    ]),
    # html.Div([
    #     html.H1("Streaming Spectrum: Demographic Threads in Topic Engagement"),
    #     html.P("Color legend: "),
    #     html.Ul([
    #         html.Li("Blue: Topics", style={'color': '#636EFA'}),
    #         html.Li("Red: Users", style={'color': '#EF553B'}),
    #     ]),
    #     generate_network_graph(df_topic_data),
    #
    # ])
    html.Div([
        html.H1("Streaming Spectrum: Demographic Threads in Topic Engagement"),
        html.Div([
            dcc.Slider(
                id='topic-identifier-slider',
                min=df_topic_data['Topic Identifier'].min(),
                max=df_topic_data['Topic Identifier'].max(),
                value=2,  # Default value
                marks={str(n): str(n) for n in range(df_topic_data['Topic Identifier'].min(), df_topic_data['Topic Identifier'].max() + 1)},
                step=1,
            )
        ]),
        html.Div([
            dcc.Dropdown(
                id='topic-label-dropdown_graph',
                options=[{'label': label, 'value': label} for label in df_topic_data['Label'].unique()],
                multi=True,
                placeholder="Filter by Topic Label",
            ),
            dcc.Dropdown(
                id='username-dropdown',
                options=[{'label': username, 'value': username} for username in df_topic_data['UserName'].unique()],
                multi=True,
                placeholder="Filter by UserName",
            ),

        ]),
        html.P("Color legend: "),
        html.Ul([
            html.Li("Blue: Topics", style={'color': '#636EFA'}),
            html.Li("Red: Users", style={'color': '#EF553B'}),
        ]),
        html.Div(id='network-graph-container'),  # Container for the network graph
        html.Button('Refresh Visual', id='refresh-button', n_clicks=0)

    ])
    ,

    html.Div([
        html.H1("Topic Engagement Over Time Across Topics"),
        html.P("Times displayed are in UTC format", style={'textAlign': 'right','color': 'grey', 'fontSize': '12px','display': 'inline-block'   }),
        html.P(" "),

        dcc.Graph(id='time-series-chart'),
    ]),
    html.Div([
        html.H1("Engagement Threads: User Roles and Messages"),
        dcc.Dropdown(
            id='topic-label-dropdown',
            options=[
                {'label': label, 'value': label}
                for label in df_topic_data['Label'].unique()
            ],
            multi=True,
            placeholder="Filter by Topic Label",
        ),
        dash_table.DataTable(
            id='details-table',
            columns=[
                {"name": "Representative Message", "id": "Original MessageContent"},
                {"name": "Username", "id": "UserName"},
                {"name": "User Role", "id": "UserRole"},
            ],
            data=[],  # Initially table is empty, it will be populated via callback
            sort_action='native',  # Enables sorting
            filter_action='native',  # Enables filtering on each column
            style_table={'height': '300px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={
                'backgroundColor': 'light-grey',
                'fontWeight': 'bold'
            },
            style_data_conditional=[  # Conditional styling for data cells
                {'if': {'column_id': 'Original MessageContent'},
                 'backgroundColor': '#EBF4FA'},
                {'if': {'column_id': 'UserName'},
                 'backgroundColor': '#EBFAEB'},
                {'if': {'column_id': 'UserRole'},
                 'backgroundColor': '#FAEBF5'},
            ],
        ),
    ])
])



