import io
import json
import torch
from torchvision import models
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, url_for, session, redirect
from models import malaria_model,breast_model,glaucoma_model
import pandas as pd
import pickle

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
    elif disease=='breast-cancer':
        return breastc_transforms(image).unsqueeze(0)

def get_prediction(image_bytes,disease):
    if disease=='malaria':
        model = malaria_model()
    elif disease=='glaucoma':
        model = glaucoma_model()
    elif disease=='breast-cancer':
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
        image_input = ['malaria','glaucoma','breast-cancer']
        disease = request.form.get('dropdown')
        session['disease'] = disease
        if disease in image_input:
            return render_template('image.html')
        elif disease == 'heart-disease':
            return render_template('heart.html')
        elif disease == 'kidney-disease':
            return render_template('kidney.html')
        else:
            return 'No Input'

@app.route('/heartdata',methods=['POST'])
def get_heartdata():
    disease = session['disease']
    model = pickle.load(open('heart-model.pkl','rb'))
    inp = []
    cp,restecg,thal,slope = [0,0,0,0],[0,0,0],[0,0,0,0],[0,0,0]
    inp.append(int(request.form.get('age')))
    inp.append(int(request.form.get('sex')))
    inp.append(int(request.form.get('trestbps')))
    inp.append(int(request.form.get('chol')))
    inp.append(int(request.form.get('fbs')))
    inp.append(int(request.form.get('thalach')))
    inp.append(int(request.form.get('exang')))
    inp.append(float(request.form.get('oldpeak')))
    inp.append(int(request.form.get('ca')))
    cp[int(request.form.get('cp'))] = 1
    inp.extend(cp)
    restecg[int(request.form.get('restecg'))] = 1
    inp.extend(restecg)
    thal[int(request.form.get('thal'))] = 1
    inp.extend(thal)
    slope[int(request.form.get('slope'))] = 1
    inp.extend(slope)
    inp = list(np.asarray(inp).reshape(1,-1))
    bclass = ['Negative','Positive']
    res = list(model.predict(inp))
    prob = list(model.predict_proba(inp))[0][res[0]]
    return render_template('result.html',disease=disease,result=bclass[res[0]],value=str(round(prob,3)*100)+"%")

@app.route('/kidneydata',methods=['POST'])
def get_kidneydata():
    disease = session['disease']
    model = pickle.load(open('kidney-model.pkl','rb'))
    inp = []
    inp.append(float(request.form.get('age')))
    inp.append(float(request.form.get('bp')))
    inp.append(float(request.form.get('sg')))
    inp.append(float(request.form.get('al')))
    inp.append(float(request.form.get('su')))
    inp.append(float(request.form.get('rbc')))
    inp.append(float(request.form.get('pc')))
    inp.append(float(request.form.get('pcc')))
    inp.append(float(request.form.get('ba')))
    inp.append(float(request.form.get('bgr')))
    inp.append(float(request.form.get('bu')))
    inp.append(float(request.form.get('sc')))
    inp.append(float(request.form.get('sod')))
    inp.append(float(request.form.get('pot')))
    inp.append(float(request.form.get('hemo')))
    inp.append(float(request.form.get('pcv')))
    inp.append(float(request.form.get('wc')))
    inp.append(float(request.form.get('rc')))
    inp.append(float(request.form.get('htn')))
    inp.append(float(request.form.get('dm')))
    inp.append(float(request.form.get('cad')))
    inp.append(float(request.form.get('appet')))
    inp.append(float(request.form.get('pe')))
    inp.append(float(request.form.get('ane')))
    inp = list(np.asarray(inp).reshape(1,-1))
    bclass = ['Negative','Positive']
    res = int(list(model.predict(inp))[0])
    prob = list(model.predict_proba(inp))[0][res]
    print(prob,res)
    return render_template('result.html',disease=disease,result=bclass[res],value=str(round(prob,3)*100)+"%")

        

@app.route('/imagedata',methods=['POST'])
def get_image():
    disease = session['disease']
    file = request.files['file']
    img_bytes = file.read()
    bclass = ['Positive','Negative']
    prob,res = get_prediction(image_bytes=img_bytes,disease=disease)
    prob,res = prob.tolist(),res.tolist()
    prob[0][0] = round(prob[0][0],4)
    if disease=='breast-cancer':
        res[0][0] = 1-res[0][0]
    return render_template('result.html',disease=disease,result=bclass[res[0][0]],value=str(round(prob[0][0],3)*100)+"%")



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