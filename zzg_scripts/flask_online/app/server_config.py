#encoding=utf8
import os


class Config:
    '''这里只定义基础的静态配置数据'''
    APP_NAME = 'onnx_class'
    MODEL_LOCAL = 'app/class_infer/onnx/resnet18-5c106cde.onnx'

    RET_CODE_DICT = {
                     0:  '成功', 
                     1: '模型推理内部错误',
                    }
    

class DevelopmentConfig(Config):
    APP_DESC = 'dev'
    
class TestingConfig(Config):
    APP_DESC = 'test'

class ProductionConfig(Config):
    APP_DESC = 'prod'
    
config = {
    "development": DevelopmentConfig,
    "test": TestingConfig,
    "production": ProductionConfig,
    "preview": ProductionConfig,
    "static": Config
}
