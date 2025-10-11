
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Role

class RoleSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Role
        load_instance = True
