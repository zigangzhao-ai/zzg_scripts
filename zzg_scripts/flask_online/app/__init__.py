#encoding=utf8
import os
import socket
import traceback
hostname = socket.gethostname()

import logging
import logging.handlers
from flask import Flask, request

global_config = {}
from app.server_config import config

import torch
import cv2
import base64
import json
import numpy as np
import requests as req
import time
from datetime import datetime

from app.class_infer.class_onnx import onnxClass
from logging.handlers import TimedRotatingFileHandler
import re

def setup_log(config_name, log_name):
    #create logger object
    logger = logging.getLogger(log_name)
    log_level = logging.INFO if config_name in ['production'] else logging.DEBUG
    #set log level
    logger.setLevel(log_level)
    #make log dir
    f_path = os.getenv('MATRIX_APPLOGS_DIR',".")
    log_path = os.path.join(f_path, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # interval-滚动周期
    # when="MIDNIGHT", interval=1-表示每天0点为更新点，每天生成一个文件
    # backupCount-表示日志保存个数
    file_handler = TimedRotatingFileHandler(filename=os.path.join(log_path, log_name), \
                                           when="MIDNIGHT", interval=1, backupCount=30)

    file_handler.suffix = "%Y-%m-%d.log"
    # extMatch是编译好正则表达式，用于匹配日志文件名后缀
    # 需要注意的是suffix和extMatch一定要匹配的上，如果不匹配，过期日志不会被删除
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
    # 定义日志输出格式
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s"
        )
    )
    logger.addHandler(file_handler)

    return logger

log_keys = [
    'process', 'pathname', 'funcName', 'lineno', 'asctime', 'message',
    'levelname'
]
log_format = lambda x: ['%({0:s})'.format(i) for i in x]
custom_format = ' '.join(log_format(log_keys))


def create_app(config_name):

    global global_config
    app = Flask(__name__)
    app.config.from_object(config[config_name])  # 加载配置

    logger = setup_log(config_name, 'onnx_class_log')
    # 动态配置的生成和加载
    global_config = dict(app.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_file = global_config['MODEL_LOCAL']
    class_net = onnxClass(model_file, device)
 
    ret_code_dict = global_config['RET_CODE_DICT']

    @app.route(rule='/healthcheck')
    def health():
        msg = {
            'serviceStatusName': 'UP',
            'data': None,
            'hostName': hostname,
            'serverTime': datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
               }
        return json.dumps(msg)

    @app.route(rule='/onnx_class', methods=['POST'])
    def onnx_class():
        ret_result = {
            'pred_out': None,
            'flag': 0,
            'use_time': 0,
            'error_msg': ret_code_dict[0],
            'error_code': 0
        }
        start_time = time.time()
        try:
        # if 1: 
            get_data = json.loads(request.data)
            img_pth = get_data['img_pth']
            # if img_url
            # response = req.get(img_url)
            # img = cv2.imdecode(np.frombuffer(response.content, np.uint8), flags=cv2.IMREAD_COLOR)
            img = cv2.imread(img_pth)
            result = class_net.onnx_infer(img)
            print('++++', result)
            end_time = time.time()
            use_time = end_time - start_time
            ret_result['pred_out'] = int(result)
            ret_result['flag'] = 0
            ret_result['use_time'] = round(use_time,2)
            ret_str = json.dumps(ret_result)
            logger.info("return json: %s", ret_str)
            print('return json:', ret_str)
            return ret_str
               
        except:      
            end_time = time.time()
            use_time = end_time - start_time
            logger.error(f"inference error !!!{traceback.format_exc()}")
            ret_result['flag'] = 1
            ret_result['use_time'] = round(use_time,2)
            ret_str = json.dumps(ret_result)
            print('return json:', ret_str)
            logger.error("return json: %s", ret_str)
            return ret_str
    
    return app
