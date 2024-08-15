import ConfigArgs as args
import os
# from SGRetrievalController.FindMatcherController import find_matches
# from SGGController.ReltrController import sgg_controller
from SGGControllerRelTR.RelTRController import sgg_controller
from FashionpediaController.inference import find_matcher
import flask
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('Datasets/VisualGenome/VG_100K', filename)

@app.route('/res-sgg/<filename>')
def upload_image(filename):
    print(filename)
    if('object+' in filename):
        print('check ok')
        return send_from_directory('Datasets/upload/' + filename.replace('.jpg', '').replace('object+', ''), filename)
    if('graph+' in filename):
        return send_from_directory('Datasets/upload/' + filename.replace('.jpg', '').replace('graph+', ''), filename)
    if('triplet+' in filename):
        return send_from_directory('Datasets/upload/' + filename.replace('.jpg', '').replace('triplet+', ''), filename)

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    # Kiểm tra xem có file trong request không
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']

    # Nếu người dùng không chọn file, browser cũng
    # submit một phần trống không có filename
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file:
        # Lưu file
        filepath = os.path.join(args.upload, file.filename)
        file.save(filepath)
        res = sgg_controller(file.filename)
        print(res)
        return jsonify(
            Data = res
        )

@app.route('/rev', methods = ['POST'])
@cross_origin()
def STS_filename_from_embed():
    data = request.get_json()
    print(data)
    query = data['triplet']
    image_name = find_matcher(query, top_k=9)
    print(image_name)
    return jsonify(
        Data = image_name
        # Status = 200, 
        # Msg = 'OK'
        ) 
    

if __name__ == "__main__":
    app.run(host="10.118.1.3",port=8009)
    # main()