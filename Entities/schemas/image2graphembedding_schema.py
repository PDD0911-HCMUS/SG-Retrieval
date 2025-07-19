
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Image2GraphEmbedding

class Image2GraphEmbeddingSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Image2GraphEmbedding
        load_instance = True
