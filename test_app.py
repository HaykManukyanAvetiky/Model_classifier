# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:15:22 2020

@author: Hayk
"""

from Net import Net, TempletLayer, device
import utils
import torch
from PIL import Image

model = torch.load('model_clasifier_v_1_0.torch', map_location=device)

#print(model)

image = Image.open( 'data/65212397_674340922990813_1189716705141394165_n.jpg')

transformed = model.transform(image).float()
transformed = transformed.unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    a = model(transformed) 
    
m_map = { 0: 'Plus-Sizel',
    1: 'Fitness ',
    2: 'Swimsuit and Lingerie',
    3: 'Commercial ',
    4: 'Glamour and Fashion' }
sorte, indices =  torch.exp(a).sort(descending=True) 

indices =  indices[sorte > 0.01]
sorte = sorte[sorte > 0.01]
sorte = sorte/sorte.sum()
sorte = sorte * 100

res = ''
for index, value in enumerate(sorte):
    index = indices[index].item()
    value = round(value.item())
    
    res+= str(value) + '% ' + m_map[index] + '\n'
    
#print(res)

image = utils.create_result(image,res)

image.show()