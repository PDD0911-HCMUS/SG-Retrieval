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
from .config_run import *
from config import *
from datetime import datetime, timedelta

class UserController:
    def __init__(self):
        pass

    @staticmethod
    def _hash_pass(raw_pass):
        hash_pass = {}
        try:
            hash = bcrypt.using(rounds=12).hash(raw_pass)
            hash_pass['hash'] = hash
            hash_pass['error'] = None
            return hash_pass
        except Exception as e:
            hash_pass['hash'] = None
            hash_pass['error'] = traceback.format_exc()
            return hash_pass

    @staticmethod
    def _verify_pass(raw_pass, hashed):
        verify = {}
        try:
            if(bcrypt.verify(raw_pass, hashed)):
                verify['verified'] = True
                verify['error'] = None
            else:
                verify['verified'] = False
                verify['error'] = None

            return verify
        except Exception as e:
            verify['verified'] = None
            verify['error'] = traceback.format_exc()

            return verify
        
    @staticmethod
    def _check_exist_user(user_name):
        check_user = {}
        try:
            USER = entity.User
            user = db.session.query(USER).filter(USER.Username == user_name).first()
            if not user:
                check_user['data'] = None
                check_user['exists'] = False
                check_user['error'] = None
            else:
                check_user['data'] = user
                check_user['exists'] = True
                check_user['error'] = None
            return check_user
        except Exception as e:
            check_user['data'] = None
            check_user['exists'] = False
            check_user['error'] = traceback.format_exc()
    
    def get_all_users(self):
        try:
            USER = entity.User
            data = db.session.query(
                USER.Username,
                USER.Fullname,
                USER.CreateAt,
                USER.Activate
            ).where(USER.Delete==False).all()

            data = UserSchema(many=True).dump(data)
            return jsonify(
                Data = data,
                Status = True, 
                Msg = "Succesfull"
            )
        except Exception as e:
            return jsonify(
                Data = None,
                Msg = traceback.format_exc(),
                Status = False
            )
        
    def create_user(self):
        pass

    def update_user(self):
        pass

    def delete_user(self):
        pass

    def login(self):
        try:
            data = request.get_json()
            user_name = data['user']
            raw_pass = data['pass']

            verified_user = self._check_exist_user(user_name)

            if(verified_user['exists'] == True): # -> Tồn tại người dùng -> kiểm tra mật khẩu
                hashed_pass = verified_user['data'].Password
                verify_pass = self._verify_pass(raw_pass=raw_pass, hashed=hashed_pass)
                if(verify_pass['verified'] == True): # -> Mật khẩu đúng
                    #TODO: Setup thông tin trả về: acccestoken, quyền, ....
                    ROLE = entity.Role
                    user = verified_user['data']
                    role = db.session.query(ROLE).filter(ROLE.ID == user.RoleID[0]).first()
                    claims = {
                        "uid": user.ID,
                        "roles": [role.RoleName]
                    }
                    access_token = create_access_token(identity=str(user.ID),
                                 additional_claims=claims,
                                 expires_delta=timedelta(minutes=access_token_time))
                    
                    respond_data = {
                        "accessToken": access_token,
                        "tokenType": ConfigAPI.token_type,
                        "expires": access_token_time,
                        "user": {
                            "id": user.ID,
                            "fullName": user.Fullname,
                            "role": [role.RoleName]
                        }
                    }

                    return jsonify(
                        Data = respond_data,
                        Msg = "",
                        Status = True
                    )
                else: # -> Mật khẩu sai
                    return jsonify(
                        Data = None,
                        Msg = login_failed,
                        Status = False
                    )
            elif(verified_user['exists'] == False and verified_user['error'] == None ): # -> Người dùng không tồn tại 
                return jsonify(
                    Data = None,
                    Msg = login_failed,
                    Status = False
                )

        except Exception as e:
            return jsonify(
                Data = None,
                Msg = traceback.format_exc(),
                Status = False
            )

    def register(self):
        try:
            data = request.get_json()
            user_name = data['userName']
            password = self._hash_pass(data['passWord'])
            full_name = data['fullName']
            ROLE = entity.Role
            role_id = db.session.query(ROLE.ID).filter(ROLE.RoleName=="admin").scalar() 

            USER = entity.User
            insert_user = USER(
                ID = uuid.uuid4(),
                Username = user_name,
                Password = str(password['hash']),
                Fullname = full_name,
                RoleID = [str(role_id)],
                CreateAt = datetime.now(),
                Activate = True
            )
            db.session.add(insert_user)
            db.session.commit()
            return jsonify(
                Data = None,
                Msg = register_sucessful,
                Status = True
            )
        except Exception as e:
            return jsonify(
                Data = None,
                Msg = traceback.format_exc(),
                Status = False
            )