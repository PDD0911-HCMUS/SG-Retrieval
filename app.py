from SGGControllerRelTR.RelTRController import sgg_api
from QueryController.QueryController import rev_api
from ObjectDescriptionV2Controller.QueryV2Controller import rev_v2_api
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


app.register_blueprint(sgg_api, url_prefix='/sgg')
app.register_blueprint(rev_api, url_prefix='/rev')
app.register_blueprint(rev_v2_api, url_prefix='/rev_v2')

if __name__ == "__main__":
    app.run(host="10.118.1.3",port=8009)
    # main()