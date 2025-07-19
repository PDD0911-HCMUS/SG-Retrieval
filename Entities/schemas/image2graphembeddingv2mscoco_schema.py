
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Image2GraphEmbeddingV2MSCOCO

class Image2GraphEmbeddingV2MSCOCOSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Image2GraphEmbeddingV2MSCOCO
        load_instance = True
