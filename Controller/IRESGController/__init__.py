from flask import Blueprint
from .IRESGController import IRESGController

rev_v2_api = Blueprint('rev_v2', __name__)
controller = IRESGController()

rev_v2_api.add_url_rule("/create_gallery", view_func=controller.create_gallery, methods=["GET"], strict_slashes=False)
rev_v2_api.add_url_rule("/retrieve", view_func=controller.retrieve, methods = ['POST'], strict_slashes=False)
rev_v2_api.add_url_rule("/images/<filename>", view_func=controller.serve_image, methods = ['GET'], strict_slashes=False)
