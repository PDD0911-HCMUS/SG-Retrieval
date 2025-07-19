
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Base

class BaseSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Base
        load_instance = True
