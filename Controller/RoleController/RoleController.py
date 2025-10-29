from Entities.schemas.user_schema import UserSchema
from config import db
from flask import request, jsonify, send_from_directory
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select, insert
import uuid
import traceback
from passlib.hash import bcrypt
from flask_jwt_extended import create_access_token
import Entities.entities as entity
# from config_run import *
from config import *
from datetime import datetime, timedelta

class RoleController:

    def create(self):
        try:
            data = request.get_json()
            role_name = data['roleName']
            ROLE = entity.Role

            role = ROLE(
                ID = uuid.uuid4(),
                RoleName = role_name,
                CreateAt = datetime.now(),
                Active = True
            )
            db.session.add(role)
            db.session.commit()

            return jsonify(
                Data = None,
                Msg = "",
                Status = True
            )

        except Exception as e:
            return jsonify(
                Data = None,
                Msg = traceback.format_exc(),
                Status = False
            )