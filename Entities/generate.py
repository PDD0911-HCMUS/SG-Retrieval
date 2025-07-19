import os
import subprocess
import inspect
import importlib
import sys

# Đảm bảo import được Entities.entities sau khi tạo
sys.path.append(os.path.abspath("."))

from config import ConfigDB

DB_URI = ConfigDB.SQLALCHEMY_DATABASE_URI
ENTITIES_PATH = "Entities/entities.py"
SCHEMA_FOLDER = "Entities/schemas"

def generate_entities():
    print("🔄 Generating entity models...")
    os.system(f"sqlacodegen {DB_URI} --outfile {ENTITIES_PATH}")
    print("✅ Entity models generated at", ENTITIES_PATH)

def generate_schemas():
    print("🔄 Generating schema files...")
    os.makedirs(SCHEMA_FOLDER, exist_ok=True)

    # Tải lại module entities
    import Entities.entities as entities

    for name, obj in inspect.getmembers(entities):
        if inspect.isclass(obj):
            class_name = obj.__name__
            schema_name = class_name + "Schema"
            file_name = class_name.lower() + "_schema.py"

            schema_code = f"""
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from Entities.entities import {class_name}

class {schema_name}(SQLAlchemyAutoSchema):
    class Meta:
        model = {class_name}
        load_instance = True
"""

            with open(os.path.join(SCHEMA_FOLDER, file_name), "w", encoding="utf-8") as f:
                f.write(schema_code)

            print(f"✅ Generated schema: {file_name}")

if __name__ == "__main__":
    generate_entities()
    generate_schemas()
    print("🎉 Done generating entities and schemas.")
