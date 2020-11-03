import io
import json
import torch
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, url_for, session, redirect
from models import malaria_model,breast_model,glaucoma_model
import pandas as pd

app = Flask(__name__)
app.secret_key = 'abc'

def transform_image(image_bytes,disease):
    malaria_transforms = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    glaucoma_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    breastc_transforms = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    if disease=='malaria':
        return malaria_transforms(image).unsqueeze(0)
    elif disease=='glaucoma':
        return glaucoma_transforms(image).unsqueeze(0)
    elif disease=='breastc':
        return breastc_transforms(image).unsqueeze(0)

def get_prediction(image_bytes,disease):
    if disease=='malaria':
        model = malaria_model()
    elif disease=='glaucoma':
        model = glaucoma_model()
    elif disease=='breastc':
        model = breast_model()
    model.load_state_dict(torch.load('{}-model.pt'.format(disease),map_location='cpu'),strict=False)
    model.eval()
    tensor = transform_image(image_bytes=image_bytes,disease=disease)
    outputs = F.softmax(model(tensor),dim=1)
    top_p,top_class = outputs.topk(1, dim = 1)
    return top_p,top_class

@app.route('/',methods=['GET','POST'])
def predict():
    session['disease'] = 'None'
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        file = request.files['file']
        disease = request.form.get('dropdown')
        img_bytes = file.read()
        bclass = ['Positive','Negative']
        prob,res = get_prediction(image_bytes=img_bytes,disease=disease)
        prob,res = prob.tolist(),res.tolist()
        prob[0][0] = round(prob[0][0],4)
        if disease=='breastc':
            res[0][0] = 1-res[0][0]
        if disease=='breastc':
            disease='Breast Cancer'
        elif disease=='malaria':
            disease='Malaria'
        else:
            disease='Glaucoma'
        session['disease'] = disease
        return render_template('result.html',disease=disease,result=bclass[res[0][0]],value=str(prob[0][0]*100)+"%")

@app.route('/treatment',methods=['GET','POST'])
def treatment():
    disease = session.get('disease', None)
    if request.method == 'GET':
        return render_template('select_city.html')
    if request.method == 'POST':
        city = request.form.get('city-dropdown')
        df = pd.read_csv('hospital.csv')
        hospital = df.loc[(df['Disease']==disease) & (df['City']==city)]['Hospital'].tolist()
        treatment = df.loc[(df['Disease']==disease) & (df['City']==city)]['Treatment'].tolist()
        return render_template('result2.html',disease=disease,city=city,
        hospital1=hospital[0],hospital2=hospital[1],hospital3=hospital[2],
        treatment1=treatment[0],treatment2=treatment[1],treatment3=treatment[2])

    


if __name__ == '__main__':
    app.run(debug=True)