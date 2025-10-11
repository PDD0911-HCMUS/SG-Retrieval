from flask import Blueprint
from .UserController import UserController

user_api = Blueprint('user', __name__)
controller = UserController()

user_api.add_url_rule("/register", view_func=controller.register, methods=["POST"], strict_slashes=False)
user_api.add_url_rule("/login", view_func=controller.login, methods=["POST"], strict_slashes=False)