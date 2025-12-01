from flask import Flask
from config import ConfigDB, ConfigApp, db
from extensions import jwt, cors
import os

# ==== Register blueprints ====
from Controller.RelTRController import sgg_api
from Controller.IRESGController import rev_v2_api
from Controller.PageController import page_api
from Controller.UserController import user_api
from Controller.RoleController import role_api
from Controller.UtilitiesController.ConnectDBController import util_api
def create_app():
    app = Flask(__name__)

    # ==== Config ====
    app.config['SQLALCHEMY_DATABASE_URI'] = ConfigDB.SQLALCHEMY_DATABASE_URI
    app.config['CORS_HEADERS'] = 'Content-Type'
    # JWT config (đặt thật sự bằng ENV ở production)
    app.config["JWT_SECRET_KEY"] = os.environ["JWT_SECRET_KEY"]
    app.config['JWT_TOKEN_LOCATION'] = ['headers']          # + 'cookies' nếu dùng refresh cookie
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 900            # 15 phút

    # ==== Init extensions ====
    db.init_app(app)
    jwt.init_app(app)
    cors.init_app(app, resources={r"/*": {"origins": "*"}})



    app.register_blueprint(sgg_api,   url_prefix='/sgg')
    app.register_blueprint(rev_v2_api,url_prefix='/rev_v2')
    app.register_blueprint(page_api,  url_prefix='/page')
    app.register_blueprint(util_api,  url_prefix='/util')
    app.register_blueprint(user_api,  url_prefix='/user')
    app.register_blueprint(role_api,  url_prefix='/role')

    return app
