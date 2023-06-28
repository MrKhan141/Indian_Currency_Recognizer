import os
import pandas as pd
import numpy as np
import gradio as gr
from pathlib import Path
from PIL import Image
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model(r"C:\Users\Nawaz\Documents\Data_Science\Project_Phase_2\currency_6.h5")

#label_dict = {"10_Front":0,"10_Back":1 ,"10_new_Front":2,"10_new_Back":3,"20_Front":4 ,"20_Back":5 ,"20_new_Front":6,"20_new_Back":7,"50_Front":8 , "50_Back":9 ,"50_new_Front":10,"50_new_Back":11,"100_Front":12 , "100_Back":13,"100_new_Front":14,"100_new_Back":15,"200_Front":16,"200_Back":17,"500_Front":18,"500_Back":19,"2000_Front":20,"2000_Back":21}
labels = ["10_Front","10_Back","10_new_Front","10_new_Back","20_Front","20_Back","20_new_Front","20_new_Back","50_Front" ,"50_Back" ,"50_new_Front","50_new_Back","100_Front" ,"100_Back","100_new_Front","100_new_Back","200_Front","200_Back","500_Front","500_Back","2000_Front","2000_Back"]
#labels = ['AMERICAN GOLDFINCH','BARN OWL','CARMINE BEE-EATER','DOWNY WOODPECKER','EMPEROR PENGUIN','FLAMINGO']
#np.format_float_positional(pred2, trim="-")
#labels=["10_Front","10_Back","20_Front","20_Back","50_Front","50_Back" ,"100_Front","100_Back"]    
def predict_note(img):     
        img = img.reshape(-1,128,128,3)
        prediction = model.predict(img/255.0).flatten()
        index = np.argmax(prediction)
        confidences = {labels[i]:float(prediction[i]) for i in range(8)}
        return (confidences)

output =[gr.Label(num_top_classes=5)]

gr.Interface(fn=predict_note, 
             inputs=gr.Image(shape=(128,128)),
             outputs=output,
            ).launch()