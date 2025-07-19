
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Date

class DateSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Date
        load_instance = True
