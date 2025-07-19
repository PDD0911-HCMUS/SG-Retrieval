
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import Identity

class IdentitySchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Identity
        load_instance = True
