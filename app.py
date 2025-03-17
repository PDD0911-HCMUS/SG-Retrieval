from RelTRSGGController.RelTRController import sgg_api
from IRESGCLController.IRESGCLController import rev_api
from flask import Flask
from flask_cors import CORS
from ConfigArgs import Config, db

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["SQLALCHEMY_DATABASE_URI"] = Config.SQLALCHEMY_DATABASE_URI
db.init_app(app)


app.register_blueprint(sgg_api, url_prefix='/sgg')
app.register_blueprint(rev_api, url_prefix='/rev')

if __name__ == "__main__":
    app.run(host="10.118.1.3",port=8009)