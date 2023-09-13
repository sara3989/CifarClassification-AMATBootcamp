import dash_bootstrap_components as dbc
from dash import dcc, html

# Constants
img_width = 700
img_height = 700
scale_factor = 0.5

video_modal = html.Div(
    dbc.Modal(
        [
            dbc.ModalHeader("Video Feed"),
            dbc.ModalBody(html.Img(id="video-feed", src="/video_feed", width=400, height=300)),
            dbc.ModalFooter(dbc.Button("Take Photo", id="take-photo-btn", color="primary", className="mr-1"), )
        ],
        id="video-modal",
        size="md",
    )
)

layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3('Welcome to image classifying app', style={'margin': '1em'}), width=5),
        dbc.Col(dcc.Upload(
            id='upload-image',
            children=dbc.Container([
                'Drag and Drop or ',
                dbc.Button('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ), width=4),
        dbc.Col(dbc.Button('Use Camera', id='open-modal-btn', style={'margin': '1.3em'}), width=3)]),
    video_modal,
    dcc.Store(id='cropped-image-store', data=None),  # Store the cropped image data
    dcc.Store(id='original-image-store', data=None),
    dbc.Row([
        dbc.Col(
            dbc.Container(id='output-image-upload'),
            width=5
        ),
        dbc.Col(
            dbc.Container([html.H3(id="predicted-class"),
                           dbc.Container(id="feedback")])
        )
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Button("Predict Image", id="predict-button", style={'margin': '10px'}),
            width=6
        ),
    ]),
], fluid=True)


def parse_contents(fig, filename):
    return dbc.Card([
        html.H5(filename),
        dcc.Graph(figure=fig,
                  config={'displayModeBar': False},  # Disable the mode bar
                  style={
                      'width': '100%',
                      'height': '100%',  # Take full height of the card
                      'padding': 0,  # Remove padding to fit the figure completely
                  }, ),
    ],
        style={
            'width': '100%',
            'height': '450px',
            'textAlign': 'center',
            'padding': '20px'
        },
    )


def feedback_content():
    content = [html.H3('How was the prediction?'),
               dbc.Button([html.I(className="bi bi-hand-thumbs-up-fill")], id="like"),
               dbc.Button([html.I(className="bi bi-hand-thumbs-down-fill")], id="dislike")]
    return content
