
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import GraphRetrievalAllMiniLML6V2

class GraphRetrievalAllMiniLML6V2Schema(SQLAlchemyAutoSchema):
    class Meta:
        model = GraphRetrievalAllMiniLML6V2
        load_instance = True
