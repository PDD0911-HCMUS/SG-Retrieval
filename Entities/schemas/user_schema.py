
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import User

class UserSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = User
        load_instance = True
