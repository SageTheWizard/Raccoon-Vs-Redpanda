import flask
from flask import request
from flask_cors import CORS

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


@app.route('/', methods=['POST'])
def home():
    data = request.get_json()

    return data['picofRacoon']


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return "Nah dawg"
    if request.method == 'POST':
        f = request.files['picofRacoon']
        # do your stuff here
        f.save('D:/Code/People/jacob/restapi/downloads/' + f.filename)

        return "Racoon?" # return the good


# app.run()
