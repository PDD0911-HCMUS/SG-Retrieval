
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Image2GraphEmbeddingV2

class Image2GraphEmbeddingV2Schema(SQLAlchemyAutoSchema):
    class Meta:
        model = Image2GraphEmbeddingV2
        load_instance = True
