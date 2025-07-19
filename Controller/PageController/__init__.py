from flask import Blueprint
from .PageController import PageController

page_api = Blueprint('page', __name__)
controller = PageController()

page_api.add_url_rule("/getall", view_func=controller.get_all_pages, methods=["GET"], strict_slashes=False)
page_api.add_url_rule("/insert", view_func=controller.insert_page, methods=["POST"], strict_slashes=False)
page_api.add_url_rule("/update", view_func=controller.update_page, methods=["PUT"], strict_slashes=False)
page_api.add_url_rule("/delete", view_func=controller.delete_page, methods=["DELETE"], strict_slashes=False)
page_api.add_url_rule("/check", view_func=controller.check_api, methods=["GET"], strict_slashes=False)