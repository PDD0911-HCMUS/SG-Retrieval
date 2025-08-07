
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import BigInteger

class BigIntegerSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = BigInteger
        load_instance = True
