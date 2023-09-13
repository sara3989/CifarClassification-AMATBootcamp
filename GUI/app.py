from dash import Dash
import dash_bootstrap_components as dbc
from GUI.layout import layout
from GUI.callbacks import register_callbacks
from GUI.server import server


app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR, dbc.icons.BOOTSTRAP], server=server, suppress_callback_exceptions=True)
app.layout = layout

register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)