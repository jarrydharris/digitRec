import os
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_canvas import DashCanvas
from dash_table import DataTable
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import json
from torch import from_numpy, load, max
from dash_canvas.utils.parse_json import parse_jsonstring
from skimage.transform import resize, rescale
from hoonn_model import Net

def predict(input_image):
    test = from_numpy((input_image - 0.1307) / 0.3081)
    test = test.unsqueeze(0)
    test = test.unsqueeze(1)

    model = Net()
    model.load_state_dict(load('saved_models\HooNN.pt'))
    model.eval()
    output = model(test.float())
    pred = max(output.data, 1)[1]
    return (pred.item())

app = dash.Dash(__name__)

server = app.server

#TODO: height is 200px, https://github.com/plotly/dash-canvas/issues/37 <- potential commit
width = 200
height = 200

app.layout = html.Div([

    DashCanvas(id='canvas_1',
    width=width,
    hide_buttons=['line', 'zoom','pan','pencil','rectangle','select'],
    lineWidth=20,
    lineColor='black'),
    html.P(id='dummy-output')

])





@app.callback(Output('dummy-output', 'children'),
              [Input('canvas_1', 'json_data')])
def update_data(string):
    if string:
        imageArray = parse_jsonstring(string)
        imageCropped = imageArray[0:width,0:height]
        rescaledImageArray = rescale(imageCropped, (0.14, 0.14))
        prediction = predict(rescaledImageArray)
    else:
        raise PreventUpdate
    return "Your number is: " + str(prediction)

if __name__ == '__main__':
    app.run_server(debug=True)