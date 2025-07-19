
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Boolean

class BooleanSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Boolean
        load_instance = True
