from config import db
from flask_cors import CORS, cross_origin
from flask import Blueprint, request, jsonify, send_from_directory
from sqlalchemy.exc import SQLAlchemyError

page_api = Blueprint('page', __name__)

@page_api.route("/get_all_pages", methods = ['GET'])
def get_all_pages():
    # TODO: Get all pages with permission
    return

@page_api.route("/update_page", methods = ['POST'])
def update_page():
    return

@page_api.route("/insert_page", methods = ['POST'])
def insert_page():
    return

@page_api.route("/delete_page", methods = ['POST'])
def delete_page():
    return