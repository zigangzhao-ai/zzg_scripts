#!/usr/bin
# Author       : zzg 
# Last modified: 2021-09-26 15:31
# Email        : 1415411655@qq.com

import os

ori_url = "https://p.ke.com/v1/raw/p3/utopia-smart-construction-site/0894e3cd890610c98b5718801620f0102801300138e02d.jpeg"
cmd = "curl utopia-smart-construction-site.lixiaojing03.ttb.test.ke.com/constructionsite/camera/get/preSigned/image/url?originUrl=" + ori_url
print(cmd)

os.system(cmd)



