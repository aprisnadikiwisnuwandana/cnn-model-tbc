from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os
import mahotas as mh

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
# model = load_model('model.h5')

labtbc = {'Normal': 1, 'Tuberculosis': 0}
labcvd = {'Covid': 0, 'Normal': 1, 'Viral Pneumonia': 2}

IMM_SIZE = 224

def diagnosiscovid(file):
    # Download image
    ##YOUR CODE GOES HERE##
    image = mh.imread(file)
    
    # Prepare image to classification
    ##YOUR CODE GOES HERE##
    if len(image.shape) > 2:
      image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]]) # resize of RGB and png images
    else:
      image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE]) # resize of grey images    
    if len(image.shape) > 2:
      image = mh.colors.rgb2grey(image[:,:,:3], dtype = np.uint8)  # change of colormap of images alpha chanel delete
    

    # Load model  
    ##YOUR CODE GOES HERE##
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into a new model
    model.load_weights("model.h5")


    # Normalize the data
    ##YOUR CODE GOES HERE##
    image = np.array(image) / 255
    
    # Reshape input images
    ##YOUR CODE GOES HERE##
    image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)

    
    # Predict the diagnosis
    ##YOUR CODE GOES HERE##
    predict_x=model.predict(image) 
    predictions=np.argmax(predict_x,axis=1)
    predictions = predictions.reshape(1,-1)[0]

    
    # Find the name of the diagnosis  
    ##YOUR CODE GOES HERE##
    diag = {key for key in labcvd if labcvd [key]==predictions}
    
    return diag

def diagnosistbc(file):
    # Download image
    ##YOUR CODE GOES HERE##
    image = mh.imread(file)
    
    # Prepare image to classification
    ##YOUR CODE GOES HERE##
    if len(image.shape) > 2:
      image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]]) # resize of RGB and png images
    else:
      image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE]) # resize of grey images    
    if len(image.shape) > 2:
      image = mh.colors.rgb2grey(image[:,:,:3], dtype = np.uint8)  # change of colormap of images alpha chanel delete
    

    # Load model  
    ##YOUR CODE GOES HERE##
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into a new model
    model.load_weights("model2.h5")


    # Normalize the data
    ##YOUR CODE GOES HERE##
    image = np.array(image) / 255
    
    # Reshape input images
    ##YOUR CODE GOES HERE##
    image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)

    
    # Predict the diagnosis
    ##YOUR CODE GOES HERE##
    predict_x=model.predict(image) 
    predictions=np.argmax(predict_x,axis=1)
    predictions = predictions.reshape(1,-1)[0]

    
    # Find the name of the diagnosis  
    ##YOUR CODE GOES HERE##
    diag = {key for key in labtbc if labtbc [key]==predictions}
    
    return diag

@app.route('/cvd', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = diagnosiscovid(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/tbc', methods=['GET', 'POST'])
def index2():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = diagnosistbc(img_path)
            return render_template('index2.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index2.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)