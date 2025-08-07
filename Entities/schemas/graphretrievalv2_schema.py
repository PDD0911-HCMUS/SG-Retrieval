
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import GraphRetrievalV2

class GraphRetrievalV2Schema(SQLAlchemyAutoSchema):
    class Meta:
        model = GraphRetrievalV2
        load_instance = True
