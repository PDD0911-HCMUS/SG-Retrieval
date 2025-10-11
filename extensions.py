# extensions.py
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from config import db

jwt = JWTManager()
cors = CORS()
