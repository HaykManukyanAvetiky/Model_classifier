# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:13:21 2020

@author: Hayk
"""

from flask import Flask, render_template, request, send_from_directory, redirect
from PIL import Image
from Net import Net, TempletLayer
import classifier
import utils
import os

app = Flask(__name__) # , static_folder='templates/img'

#image = Image.open( 'test_data/80e599ad042eb77aefb07103a243a1d5.jpg')
#image = classifier.predict(image)

#image.save(path)



@app.route('/', methods=["GET", "POST"])
@app.route('/index', methods=["GET", "POST"])
def home():
    path = "img/types-of-female-models.png"
    if request.method == "POST":
        if request.files:    
            image = Image.open( request.files["image"]  ) 
            image = classifier.predict(image)
            #path = os.path.join( app.static_folder, "/SYS_TEMP", utils.get_path())            
            path = utils.get_path()
            image.save(os.path.join('static', path))
            #return redirect(request.url)
            return render_template('index.jinja2',path=path)
            
        
    return render_template('index.jinja2',path=path)
    


if __name__ == '__main__':
	app.run(debug = True)
    
    