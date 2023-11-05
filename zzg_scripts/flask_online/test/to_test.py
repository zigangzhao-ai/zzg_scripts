import os
import io
import cv2
import json
import time
import base64
import requests

import numpy as np
from PIL import Image
#from requests_toolbelt.multipart.encoder import MultipartEncoder
import json

def test_image():
 
    app_url = 'http://10.100.35.83:8081/onnx_class'
    img_pth = "/code/zzg/project/study/flask_online/test/cab.png"

    infos = {'img_pth': img_pth} 
            
    headers = {'Content-Type': 'application/json'}
    datas = json.dumps(infos)
    res = requests.post(url=app_url, data=datas, headers=headers)
    print('----', res.content)
    result = json.loads(res.content)
    for key in result:
        print(key, result[key])

        
if __name__ == '__main__':

    test_image()

    print('done!')



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
