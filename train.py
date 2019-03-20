# -*- coding: UTF-8 -*-
import os
import sys



#设置路径信息
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'utils'))
sys.path.append(os.path.join(BASE_DIR,'models'))
import provider