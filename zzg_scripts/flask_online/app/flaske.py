#encoding=utf8
import os
from app import create_app

app = create_app(os.getenv('ENVTYPE') or 'test')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
