#encoding=utf8
import json
import logging
from flask import Response, request
from flask_restful import Api, Resource, reqparse
from app.demo.argu_capsulation import reqparser
from app.demo.response_capsulation import server_code_info


class Base:
    def __init__(self):
        self.args = reqparser.parse_args()

class HealthCheck(Resource, Base):
    def __init__(self):
        Base.__init__(self)
        self.result = {
            "status": 0,
            "data": {},
            "msg": "状态的描述",
            "debug_data": {}
        }

    def get(self):
        status = 200
        self.result['msg'] = server_code_info.get(status)
        self.result['code'] = 0
        return Response(
            response=json.dumps(self.result, ensure_ascii=False),
            status=status,
            headers={'Content-Type': 'application/json;charset=UTF-8'})
