from Controller.RelTRSGGController import sgg_api
from Controller.IRESGCLController.IRESGCLController import rev_api
from Controller.IRESGController import rev_v2_api
from Controller.PageController import page_api
from Controller.UtilitiesController.ConnectDBController import util_api 
from flask import Flask
from flask_cors import CORS
from config import ConfigDB, db, ConfigApp

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["SQLALCHEMY_DATABASE_URI"] = ConfigDB.SQLALCHEMY_DATABASE_URI
db.init_app(app)


app.register_blueprint(sgg_api, url_prefix='/sgg')
app.register_blueprint(rev_api, url_prefix='/rev')
app.register_blueprint(rev_v2_api, url_prefix='/rev_v2')
app.register_blueprint(page_api, url_prefix='/page')
app.register_blueprint(util_api, url_prefix='/util')

if __name__ == "__main__":
    # create_gallery()
    app.run(host=ConfigApp.domain,port=ConfigApp.port)