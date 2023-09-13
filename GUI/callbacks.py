from dash import Output, Input, State, no_update, callback_context
from GUI.utils import VideoCamera, image_to_base64, base64_to_image, predict, user_feedback, classes
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from GUI.layout import img_width, img_height, scale_factor, parse_contents, feedback_content
import plotly.io as pio
from PIL import Image
import base64
import io
from dash import dcc, html
import dash_bootstrap_components as dbc

def register_callbacks(app):
    @app.callback(
        Output('output-image-upload', 'children'),
        Input('upload-image', 'contents'),
        Input("take-photo-btn", "n_clicks"),
        State('upload-image', 'filename')
    )
    def update_output(list_of_contents, take_photo_clicks, list_of_names):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Handle file upload
        if trigger_id == 'upload-image':
            fig = go.Figure()

            # Configure axes
            fig.update_xaxes(
                visible=False,
                range=[0, img_width * scale_factor]
            )

            fig.update_yaxes(
                visible=False,
                range=[0, img_height * scale_factor],
                scaleanchor="x"
            )

            if list_of_contents:
                fig.add_layout_image(
                    dict(
                        x=0,
                        sizex=img_width * scale_factor,
                        y=img_height * scale_factor,
                        sizey=img_height * scale_factor,
                        xref="x",
                        yref="y",
                        opacity=1.0,
                        layer="below",
                        source=list_of_contents)
                )

            fig.update_layout(
                width=img_width * scale_factor,
                height=img_height * scale_factor,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
            )

            children = [parse_contents(fig, list_of_names)]

            return children

        # Handle taking a photo from the camera
        elif trigger_id == 'take-photo-btn':
            if take_photo_clicks is None or take_photo_clicks == 0:
                raise PreventUpdate

            # Take a photo from the camera
            camera = VideoCamera()
            frame = camera.get_frame()
            fig = go.Figure()

            # Configure axes
            fig.update_xaxes(
                visible=False,
                range=[0, img_width * scale_factor]
            )

            fig.update_yaxes(
                visible=False,
                range=[0, img_height * scale_factor],
                scaleanchor="x"
            )

            fig.add_layout_image(
                dict(
                    x=0,
                    sizex=img_width * scale_factor,
                    y=img_height * scale_factor,
                    sizey=img_height * scale_factor,
                    xref="x",
                    yref="y",
                    opacity=1.0,
                    layer="below",
                    source="data:image/jpeg;base64," + base64.b64encode(frame).decode("utf-8"))
            )

            fig.update_layout(
                width=img_width * scale_factor,
                height=img_height * scale_factor,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
            )

            children = [parse_contents(fig, 'from webcam')]
            return children

        return no_update

    @app.callback(
        Output("predicted-class", "children"),
        Output("feedback", "children", allow_duplicate=True),
        Output("cropped-image-store", "data"),
        Input("predict-button", "n_clicks"),
        State("output-image-upload", "children"),
        prevent_initial_call=True
    )
    def predict_image(n_clicks, children):
        if n_clicks and children:
            original_figure = children[0]['props']['children'][1]['props']['figure']
            image_bytes = pio.to_image(original_figure, format="png")
            image = Image.open(io.BytesIO(image_bytes))
            predicted_class = predict(image)
            cropped_image_base64 = image_to_base64(image)
            return f"Predicted Class: {predicted_class}", feedback_content(), cropped_image_base64
        return no_update

    @app.callback(
        Output("video-modal", "is_open"),
        Output("take-photo-btn", "n_clicks"),
        Input("open-modal-btn", "n_clicks"),
        Input("take-photo-btn", "n_clicks"),
        State("video-modal", "is_open"),
    )
    def toggle_modal(open_clicks, take_photo_clicks, is_open):
        ctx = callback_context
        if not ctx.triggered:
            return False, 0
        prop_id = ctx.triggered[0]["prop_id"]
        if "open-modal-btn" in prop_id:
            return not is_open, 0
        elif "take-photo-btn" in prop_id and is_open:
            return not is_open, take_photo_clicks + 1
        raise PreventUpdate

    @app.callback(
        Output("feedback", "children", allow_duplicate=True),
        Input("like", "n_clicks"),
        Input("dislike", "n_clicks"),
        State("cropped-image-store", "data"),
        prevent_initial_call=True
    )
    def handle_feedback(like_clicks, dislike_clicks, cropped_image_data):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]["prop_id"]

        if "like" in prop_id and like_clicks:
            if cropped_image_data:
                image = base64_to_image(cropped_image_data)
                user_feedback(image, True)
                return html.H3('Thanks for your feedback')

        if "dislike" in prop_id and dislike_clicks:
            return [
                html.H3('Please choose the correct class'),
                dcc.Dropdown(
                    id="feedback_dropdown",
                    options=[{'label': x, 'value': x} for x in classes],
                    placeholder="Select a class",
                )
            ]

        return no_update
    @app.callback(
    Output("feedback", "children", allow_duplicate=True),
    Input("feedback_dropdown", "value"),
    State("cropped-image-store", "data"),
    prevent_initial_call=True
    )
    def correct_class(value, image_data):
        if value and image_data:
            image = base64_to_image(image_data)
            user_feedback(image, False, value)
            return html.H3('Thanks for your feedback')
        return no_update