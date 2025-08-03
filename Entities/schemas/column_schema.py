
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Column

class ColumnSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Column
        load_instance = True
