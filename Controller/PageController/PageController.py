from config import db
from flask_cors import CORS, cross_origin
from flask import Blueprint, request, jsonify, send_from_directory
from sqlalchemy.exc import SQLAlchemyError
import uuid
import Entities.entities as entity
from Entities.schemas.pages_schema import PagesSchema
import traceback

class PageController:
    def get_all_pages(self):
        # TODO: Get all pages with permission
        try:
            PAGE = entity.Pages
            data = db.session.query(
                PAGE.ID,
                PAGE.PageName,
                PAGE.PageURL,
                PAGE.PageLogo,
                PAGE.Activate
            ).where(PAGE.Delete==False).all()

            data = PagesSchema(many=True).dump(data)

            return jsonify(
                Data = data,
                Status = True, 
                Msg = "Succesfull"
            )
        except Exception as e:
            data = None
            msg = traceback.format_exc()
            status = False
            return jsonify(
                Data = data,
                Msg = msg,
                Status = status
            )
    def update_page(self):
        return

    def insert_page(self):
        return

    def delete_page(self):
        return
    
    def check_api(self):
        return jsonify(
            Data = None,
            Status = False, 
            Msg = "Done"
        )