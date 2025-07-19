
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import TestTable

class TestTableSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = TestTable
        load_instance = True
