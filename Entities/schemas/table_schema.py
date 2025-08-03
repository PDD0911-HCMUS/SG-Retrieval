
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Table

class TableSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Table
        load_instance = True
