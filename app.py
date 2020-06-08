# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:13:21 2020

@author: Hayk
"""

from flask import Flask, render_template, request
from PIL import Image
from Net import Net, TempletLayer
import classifier

app = Flask(__name__, static_folder='templates/img')

#image = Image.open( 'test_data/images (1).jfif')

#model.predict(image).show()

@app.route('/')
@app.route('/index')
def home():
    return render_template('index.jinja2')

@app.route('/result')
def result():
    image = request.args.get('image')
    image.show()
    return render_template('result.jinja2')




if __name__ == '__main__':
	app.run(debug = True)