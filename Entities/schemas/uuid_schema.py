
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Uuid

class UuidSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Uuid
        load_instance = True
