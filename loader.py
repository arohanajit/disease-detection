import os

def load_data():
    os.system('wget https://raw.githubusercontent.com/arohanajit/disease-detection/main/kaggle.json --no-check-certificate')
    os.system('pip install -q kaggle')
    os.system('mkdir -p ~/.kaggle')
    os.system('cp kaggle.json ~/.kaggle/')
    os.system('ls ~/.kaggle')
    os.system('chmod 600 /root/.kaggle/kaggle.json')

