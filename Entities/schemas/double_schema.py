
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Double

class DoubleSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Double
        load_instance = True
