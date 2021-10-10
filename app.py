#import os
import cv2
import mahotas as mt
#from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import pyrebase
#import joblib



config = {
	"apiKey": "AIzaSyDbGBvceqiGUB-iPJN7AXXS-AyUx80cg6s",
    "authDomain": "esp32ndvi.firebaseapp.com",
    "databaseURL": "https://esp32ndvi-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "esp32ndvi",
    "storageBucket": "esp32ndvi.appspot.com",
    "messagingSenderId": "960081609737",
    "appId": "1:960081609737:web:bc2cc955d5b7f1737be4d8",
    "measurementId": "G-9LZKTS2PPH"
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()

app = Flask(__name__)

i = 0

@app.route('/ndvi', methods=['GET','POST'])
def basic():
	if request.method == 'POST':
		if request.form['submit'] == 'add':
                    ndvis = db.child("Sensor").get()
                    name = request.form['name']
                    for ndvi in ndvis.each():
                            while (ndvi.key() == name):
                                ndvi = db.child("Sensor").child(name).get()
                                to = ndvi.val()
                                print (ndvi.key())
                                return render_template('index.html', t=to.values())

	return render_template('index.html', mess = "Please enter right device ID")


@app.route('/home')
def ss():
	    return render_template('admin.html')

@app.route("/")
def template_test():
        return render_template('search.html')




if __name__ == '__main__':
    app.run(debug=True)