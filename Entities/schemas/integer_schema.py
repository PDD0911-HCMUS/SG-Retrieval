
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Integer

class IntegerSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Integer
        load_instance = True
