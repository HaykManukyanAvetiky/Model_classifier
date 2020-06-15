# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:12:25 2020

@author: Hayk
"""


import os
import time 
import glob


while True:
    files = glob.glob('static/img/chache/*')
    for f in files:
        os.remove(f)
    time.sleep(300)


