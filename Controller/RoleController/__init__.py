from flask import Blueprint
from .RoleController import RoleController

role_api = Blueprint('role', __name__)
controller = RoleController()

role_api.add_url_rule("/create", view_func=controller.create, methods=["POST"], strict_slashes=False)