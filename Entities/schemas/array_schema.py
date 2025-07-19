
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import ARRAY

class ARRAYSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = ARRAY
        load_instance = True
