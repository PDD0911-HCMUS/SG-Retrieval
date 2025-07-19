
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Mapped

class MappedSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Mapped
        load_instance = True
