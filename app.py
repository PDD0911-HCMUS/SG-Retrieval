

import ConfigArgs as args
from SGRetrievalController.FindMatcherController import find_matches
import flask
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('Datasets/VisualGenome/VG_100K', filename)

@app.route('/rev/<user_input>', methods = ['GET'])
@cross_origin()
def STS_filename_from_embed(user_input):
    print(user_input)
    query = user_input
    indices, image_name = find_matches(query, k=9, normalize=True)
    selected_files = [image_name[i] for i in indices[0].tolist()]
    return jsonify(
        Data = selected_files
        # Status = 200, 
        # Msg = 'OK'
        ) 
    

if __name__ == "__main__":
    app.run(host="10.118.1.3",port=8009)
    # main()