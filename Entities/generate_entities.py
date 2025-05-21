import os
import config as args

# DB_URI = "postgresql://postgres:123456@localhost:5432/RetrievalSystemTraffic"
DB_URI = args.ConfigDB.SQLALCHEMY_DATABASE_URI

try:

    os.makedirs("Entities", exist_ok=True)

    # Run sqlacodegen command to generate file models
    os.system(f"sqlacodegen {DB_URI} --outfile Entities/entities.py")
    print("✅ Entity models were generated at Entities/entities.py")

except Exception as e:
    print(f"{e}")