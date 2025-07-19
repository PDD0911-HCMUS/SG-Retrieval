
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Text

class TextSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Text
        load_instance = True
