import json
import psycopg2
import config as args
from flask_cors import CORS, cross_origin
from flask import Blueprint, request, jsonify, send_from_directory
import torch


util_api = Blueprint('util', __name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@util_api.route('/checkdb', methods = ['GET'])
@cross_origin()
def check_db_connection():
    try:
        conn = psycopg2.connect(args.conn_str)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        record = cursor.fetchone()

        statusConn = True
        status = True
        statusConnMess = f'Connected to the database. You are connected to - {record}'
    except Exception as e:
        statusConn = False
        statusConnMess = f'Connect to the database failed. Error: {e}'
        status = False
    finally:
        if conn is not None:
            conn.close()
    return jsonify(
        Data = statusConn,
        Msg = statusConnMess,
        Status = status
    ) 