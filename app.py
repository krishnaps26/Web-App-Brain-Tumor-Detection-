import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
#it will return a secure version of it. This filename can then safely be stored on a regular file system and passed to os. path.
from werkzeug.utils import secure_filename  

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'D:/bt project/temp/temp/brain-tumour-webapp-main/models/modelres50.h5'

#Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(200,200)) #target size must be same as the size mentioned in the model 

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255  #feature scaling : done to normalize the range of the features present in the dataset
   
    preds = model.predict(img) # returns array of label of each object in data with their respective probabilty distribution value 
    pred = np.argmax(preds,axis = 1) #Select the result category with highest probability value
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pred = model_predict(file_path, model)
        os.remove(file_path)#removes file from the server after prediction has been returned

        str0 = 'Glioma'
        str1 = 'Meningioma'
        str3 = 'pituitary'
        str2 = 'No Tumour'
        if pred[0] == 0:
            return str0
        elif pred[0] == 1:
            return str1
        elif pred[0]==3:
            return str3
        else:
            return str2
    return None

if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=5000)    #debug is true for the development phase
    
