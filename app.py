import os
import cv2
import mahotas as mt
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import pyrebase
import joblib



def bg_sub(file):
    main_img  = cv2.imread(file)
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(img, (1400, 1000))
    size_y,size_x,_ = img.shape
    gr_scale = cv2.cvtColor(resized_image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gr_scale, (5,5),0)
    ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((50,50),np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
    
    contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    contains = []
    y_ri,x_ri, _ = resized_image.shape
    for cc in contours:
        yn = cv2.pointPolygonTest(cc,(x_ri//2,y_ri//2),False)
        contains.append(yn)

    val = [contains.index(temp) for temp in contains if temp>0]
    index = val[0]
    
    black_img = np.empty([1000,1400,3],dtype=np.uint8)
    black_img.fill(0)
    
    cnt = contours[index]
    mask = cv2.drawContours(black_img, [cnt] , 0, (255,255,255), -1)
    
    maskedImg = cv2.bitwise_and(resized_image, mask)
    white_pix = [255,255,255]
    black_pix = [0,0,0]
    
    final_img = maskedImg
    h,w,channels = final_img.shape
    for x in range(0,w):
        for y in range(0,h):
            channels_xy = final_img[y,x]
            if all(channels_xy == black_pix):
                final_img[y,x] = white_pix
    return final_img

def feature_extract(img):  

    #Preprocessing
    gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25,25),0)
    ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((50,50),np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    #Shape features
    contours, _ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    rectangularity = w*h/area
    circularity = ((perimeter)**2)/area

    #Color features
    red_channel = img[:,:,0]
    green_channel = img[:,:,1]
    blue_channel = img[:,:,2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)

    #Texture features
    textures = mt.features.haralick(gs)
    ht_mean = textures.mean(axis=0)
    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]

    vector = [area,perimeter,w,h,aspect_ratio,rectangularity,circularity,\
              red_mean,green_mean,blue_mean,red_std,green_std,blue_std,\
              contrast,correlation,inverse_diff_moments,entropy
             ]

    df_temp = pd.DataFrame([vector])

    print(vector)
    
    print(df_temp)

    return df_temp           
#

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


@app.route('/', methods=['GET','POST'])
def upload_file():
    #if request.method == 'POST':
       # file = request.files['file']
        
       # if file and allowed_file(file.filename):
       #     filename = secure_filename(file.filename)
       #     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
       #     file.save(file_path)
       #     bg_remove_img = bg_sub(file_path)
       #     featuresOfimg = feature_extract(bg_remove_img)
       #     scaled_feature = sc_X.transform(featuresOfimg)
       #     print(scaled_feature)
            #output = model.predict_proba(scaled_feature)[0]
       #     out = {'Organic:': output[0], 'Inorganic': output[1]}
    return render_template("search.html")



if __name__ == '__main__':
    app.run(debug=True)