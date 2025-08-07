
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import IRESGVGV2

class IRESGVGV2Schema(SQLAlchemyAutoSchema):
    class Meta:
        model = IRESGVGV2
        load_instance = True
