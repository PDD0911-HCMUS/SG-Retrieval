from flask import Blueprint
from .RelTRController import RelTRController

sgg_api = Blueprint('sgg', __name__)
controller = RelTRController()

sgg_api.add_url_rule('/sgg-gen', view_func=controller.sgg_controller, methods=['POST'], strict_slashes=False)
sgg_api.add_url_rule('/res-sgg/<filename>', view_func=controller.upload_image, methods=['GET'], strict_slashes=False)