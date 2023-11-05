#encoding=utf8
from flask import Blueprint
from flask_restful import Resource, Api

hc = Blueprint('healthcheck', __name__, url_prefix='')
hc_api = Api(hc)

from .api import HealthCheck
# 通过蓝图引入的路由定义

hc_api.add_resource(
    HealthCheck,
    '/healthcheck')
