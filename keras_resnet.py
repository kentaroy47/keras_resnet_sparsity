# -*- coding: utf-8 -*-

from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
model = ResNet50(weights='imagenet')
from keras import backend as K

img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))


model.summary()
weightlist=[]
actlist=[]
activations_weights=[]
activations_weights_std=[]
activations=[]
activations_std=[]
input_shape=(224,224,3)

print("get weight data")

for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays
    if weights: # weights is not empty
        if np.size(weights)==2: # its conv or fc layer
            weightlist.append(weights)
            weights[0]=weights[0]/np.max(weights[0])
            activations_weights.append(np.mean(np.abs(weights[0])))
            activations_weights_std.append(np.std(weights[0]))

print("get input data")
            
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

# put in random input to nn
test = np.random.random(input_shape)[np.newaxis,...]
layer_outs = [func([test, 1.]) for func in functors]
              
#for i in range(len(layer_outs)):
#    layer_outs[i]=layer_outs[i]/np.max(layer_outs[i])
#    activations.append(np.mean(np.abs(layer_outs[i])))
#    activations_std.append(np.std(layer_outs[i]))

print("get sparcity")
    
import numpy as np
sparcity=[]
sparcity8b=[]
layer_size=[]

for i in range(len(layer_outs[:])):

    a=layer_outs[i]
    a=a[0].reshape(-1)
    sparcity.append((len(a)-np.count_nonzero(a))/len(a)*100)
    layer_size.append(len(a))
    
    print("zeros:", sparcity[i] )
    
    cc=np.sum(np.abs(a)<1/128) #count number of elements < 128   
    sparcity8b.append(cc/len(a)*100)
    
    print("zeros in 8b:", sparcity8b[i])

sparcity_fin=[]
layer_size_fin=[]
for i in range(len(sparcity[:])):
    if sparcity[i] > 20:
        sparcity_fin.append(sparcity8b[i])
        layer_size_fin.append(layer_size[i])
        
print("cost for adaptive processor")

cost=[]

for i in range(len(sparcity_fin[:])):
    if sparcity_fin[i] < 50:
       cost.append(50)
    else:
       cost.append(sparcity_fin[i])
           
       



        
