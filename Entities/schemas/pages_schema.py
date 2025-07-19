
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Pages

class PagesSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Pages
        load_instance = True
