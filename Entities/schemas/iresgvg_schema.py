
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import IRESGVG

class IRESGVGSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = IRESGVG
        load_instance = True
