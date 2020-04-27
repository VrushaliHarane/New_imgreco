import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from dataset import AttributesDataset, mean, std
import matplotlib.pyplot as plt
import numpy as np
from model import MultiOutputModel
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

MODEL_PATH = 'fashion.model'
LABEL_PATH = 'mlb.pickle'
#enter path to input image
DATASET_PATH = '/home/vrushali/download (2).jpeg'

image = cv2.imread(DATASET_PATH)
output = imutils.resize(image, width=400)
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


model = load_model(MODEL_PATH)
mlb = pickle.loads(open(LABEL_PATH, "rb").read())

proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

for (i, j) in enumerate(idxs):
	label = "{}".format(mlb.classes_[j])
	print(label)

if __name__ == '__main__':

    checkpoint = '/checkpoint-000050.pth'
    attributes_file = '/styles.csv'
    device = 'cpu'
    #imagg=open(args.inputimage())
    attributes = AttributesDataset(attributes_file)
    model = MultiOutputModel(n_color_classes=attributes.num_colors, n_gender_classes=attributes.num_genders,
                             n_article_classes=attributes.num_articles).to(device)
    #enter path to input image
    res= DATASET_PATH

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img = Image.open(res)
    img = val_transform(img)
    img = img.view(-1, 3, img.shape[0], img.shape[1])
    name = checkpoint
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    #checkpoint_load(model, checkpoint=args.checkpoint)  
    model.eval()    
    with torch.no_grad():
        output = model(img.to(device))
        _, predicted_gender = output['gender'].cpu().max(1)
        
        predicted_genders = attributes.gender_id_to_name[predicted_gender[0].item()]
        
        print(predicted_genders)
        

    







