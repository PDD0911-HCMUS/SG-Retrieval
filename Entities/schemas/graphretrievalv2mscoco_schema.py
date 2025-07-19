
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import GraphRetrievalV2MSCOCO

class GraphRetrievalV2MSCOCOSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = GraphRetrievalV2MSCOCO
        load_instance = True
