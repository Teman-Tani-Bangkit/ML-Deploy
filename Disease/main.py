import os

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
from skimage import io
from skimage.transform import resize
import requests as req
from PIL import Image
from io import BytesIO


app = Flask(__name__)

#Load all model
model_cassava = load_model('model/Cassava_Densenet121-0.81val.h5')
model_corn = load_model('model/Corn_Densenet121-0.95val.h5')
model_potato = load_model('model/Potato_Densenet121-0.98val.h5')
model_rice = load_model('model/Rice_Densenet121-0.84val.h5')
model_tomato = load_model('model/Tomato_Densenet121-0.97val.h5')
model_chili = load_model('model/Chili_Densenet121-0.80val.h5')

def normalize_image(image):
    #Normalize image for Rice, Cassava plant
    height, width, _ = image.shape
    norm_size = max(height, width)
    add_h = norm_size - height
    add_w = norm_size - width
    start_add_h = add_h // 2
    start_add_w = add_w // 2
    new_img = np.full((norm_size, norm_size, 3), 255, dtype=np.uint8)
    new_img[start_add_h:start_add_h + height, start_add_w:start_add_w + width, :] = image
    return new_img

def preprocessing(image):
  #Preprocess image for Cassava and Rice
  img = normalize_image(image)
  img = cv2.resize(img, (255, 255), interpolation = cv2.INTER_AREA)
  return np.array(img)

def load_image_from_url(url, target_size):
    #Load image for Tomato, Corn, Potato, Chili
    response = req.get(url)
    image = Image.open(BytesIO(response.content))
    image = image.resize(target_size)
    return image
     

@app.route('/', methods=['POST'])
def predict_disease():
    if request.method == 'POST':
        plant = request.get_json()['plant']
        if plant == 'Cassava':
            link = request.get_json()['link']
            labels = ({'cbb': 0, 'cbsd': 1, 'cgm': 2, 'cmd': 3, 'healthy': 4})
            labels = dict((v,k) for k,v in labels.items()) 
            
            image = io.imread(link)
            image = preprocessing(image)
            resized_image = resize(image, (255,255), anti_aliasing=True)
            input_images = np.reshape(resized_image, (-1, 255, 255, 3)) 
            preds = model_cassava.predict(input_images)
            label = labels[np.argmax(preds, axis=1)[0]]
            return jsonify({'result': label})
        
        elif plant == 'Rice':
            link = request.get_json()['link']
            labels = ({'BrownSpot': 0, 'Healthy': 1, 'Hispa': 2, 'LeafBlast': 3})
            labels = dict((v,k) for k,v in labels.items()) 

            image = io.imread(link)
            image = preprocessing(image)
            resized_image = resize(image, (255,255), anti_aliasing=True)
            input_images = np.reshape(resized_image, (-1, 255, 255, 3)) 
            preds = model_rice.predict(input_images)
            label = labels[np.argmax(preds, axis=1)[0]]
            return jsonify({'result': label})
        
        elif plant == 'Corn':
            link = request.get_json()['link']
            labels = ({'Corn Common rust': 0,
            'Corn Gray leaf spot': 1,
            'Corn Healthy': 2,
            'Corn Northern Leaf Blight': 3})
            labels = dict((v,k) for k,v in labels.items())

            img = load_image_from_url(link, target_size=(255,255))
            target = img_to_array(img)
            target = np.expand_dims(target, axis=0)
            target = np.vstack([target])
            target /= 255
            preds = model_corn.predict(target)
            label = labels[np.argmax(preds, axis=1)[0]]
            return jsonify({'result': label})
        
        elif plant == 'Potato':
            link = request.get_json()['link']
            labels = ({'Early Blight': 0, 'Healthy': 1, 'Late Blight': 2})
            labels = dict((v,k) for k,v in labels.items())

            img = load_image_from_url(link, target_size=(255,255))
            target = img_to_array(img)
            target = np.expand_dims(target, axis=0)
            target = np.vstack([target])
            target /= 255
            preds = model_potato.predict(target)
            label = labels[np.argmax(preds, axis=1)[0]]
            return jsonify({'result': label})
        
        elif plant == 'Tomato':
            link = request.get_json()['link']
            labels = ({'Bacterial Spot': 0,
		 'Early Blight': 1,
		 'Healthy': 2,
		 'Late Blight': 3,
		 'Leaf Mold': 4,
		 'Mosaic Virus': 5,
		 'Septoria Leaf Spot': 6,
		 'Spider Mites Two Spotted Spider Mite': 7,
		 'Target Spot': 8,
		 'Yellow Leaf Curl Virus': 9})
            labels = dict((v,k) for k,v in labels.items())

            img = load_image_from_url(link, target_size=(255,255))
            target = img_to_array(img)
            target = np.expand_dims(target, axis=0)
            target = np.vstack([target])
            target /= 255
            preds = model_tomato.predict(target)
            label = labels[np.argmax(preds, axis=1)[0]]
            return jsonify({'result': label})
        
        elif plant == 'Chili':
            link = request.get_json()['link']
            labels = ({'healthy': 0, 'leaf curl': 1, 'leaf spot': 2, 'whitefly': 3, 'yellowish': 4})
            labels = dict((v,k) for k,v in labels.items())

            img = load_image_from_url(link, target_size=(255,255))
            target = img_to_array(img)
            target = np.expand_dims(target, axis=0)
            target = np.vstack([target])
            target /= 255
            preds = model_tomato.predict(target)
            label = labels[np.argmax(preds, axis=1)[0]]
            return jsonify({'result': label})

if __name__ == '__main__':
    app.run(debug=True)


