
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import GraphRetrieval

class GraphRetrievalSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = GraphRetrieval
        load_instance = True
