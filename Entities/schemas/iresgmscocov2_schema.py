
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import IRESGMSCOCOV2

class IRESGMSCOCOV2Schema(SQLAlchemyAutoSchema):
    class Meta:
        model = IRESGMSCOCOV2
        load_instance = True
