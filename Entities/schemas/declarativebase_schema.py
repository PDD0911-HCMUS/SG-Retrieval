
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import DeclarativeBase

class DeclarativeBaseSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = DeclarativeBase
        load_instance = True
