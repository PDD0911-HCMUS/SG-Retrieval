# wsgi.py
from app_factory import create_app
from config import ConfigApp

app = create_app()

if __name__ == "__main__":
    app.run(host=ConfigApp.domain, port=ConfigApp.port)
