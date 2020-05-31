# -*- coding: utf-8 -*-
"""
Functions necessary for image procesing
Created on Wed May 20 15:29:56 2020

@author: Hayk
"""

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


def gold_frame(image):
    backgrd = Image.open( 'gold_backgrd.jpg')
    a, b = image.size
    backgrd = backgrd.resize((a+20, b+20))
    backgrd.paste(image,(10,10))
    ration = max((a,b))/400
    backgrd = backgrd.resize((int(a/ration), int(b/ration)))
    return backgrd


def add_right_pan(backgrd):
    a, b = backgrd.size
    main_img = Image.new('RGB', size=((a+50)*2, b+50 ))
    main_img.paste(backgrd, (25,25))
    return main_img



def write_on_img(image, msg):
    a, b = image.size
    draw = ImageDraw.Draw(image)
    fsize = a//30
    font = ImageFont.truetype("fonts/CaviarDreams_BoldItalic.ttf", fsize)
    draw.text(xy=(a//2, b//4), text=msg, fill=(212,175,55), font=font,spacing=10 )
    return image
        

def create_result(image, msg):
    image = gold_frame(image) 
    image = add_right_pan(image)
    image = write_on_img(image, msg)
    return image
    