
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import PrimaryKeyConstraint

class PrimaryKeyConstraintSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = PrimaryKeyConstraint
        load_instance = True
