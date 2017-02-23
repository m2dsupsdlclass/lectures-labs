from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
import h5py
import numpy as np
from scipy.misc import imread, imresize
from lxml import etree
import os

annotations = []

# Parse the xml annotation file and retrieve the path to image, its size and annotations
def extract_xml_annotation(filename):
    z = etree.parse(filename)
    objects = z.findall("/object")
    size = (int(z.find("//width").text), int(z.find("//height").text))
    fname = z.find("/filename").text
    dics = [{obj.find("name").text:[int(obj.find("bndbox/xmin").text), 
                                    int(obj.find("bndbox/ymin").text), 
                                    int(obj.find("bndbox/xmax").text), 
                                    int(obj.find("bndbox/ymax").text)]} 
            for obj in objects]
    output = {"size": size, "filename":fname, "objects":dics}
    return output

# Filters annotations keeping only those we are interested in
filters = ["dog", "cat"]
for file in os.listdir("VOCdevkit/VOC2007/Annotations/"):
    annotation = extract_xml_annotation("VOCdevkit/VOC2007/Annotations/" +file)
    new_objects = []
    for obj in annotation["objects"]:
        if list(obj.keys())[0] in filters:
            new_objects.append(obj)
    if len(new_objects)>0:
        annotation["objects"] = new_objects
        annotations.append(annotation)

# Predict batch function
def predict_batch(model, img_batch_path, img_size=None):
    img_list = []

    for im_path in img_batch_path:
        img = imread(im_path)
        if img_size:
            img = imresize(img,img_size)

        img = img.astype('float32')
        img_list.append(img)
    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')

    batch = preprocess_input(img_batch)
    return model.predict(x = img_batch)

# Build the resnet
model = ResNet50(include_top=False)
input = model.layers[0].input

# Remove the average pooling layer!
output = model.layers[-2].output
headless_conv = Model(input = input, output = output)

# Computing representations for all images
def compute_representations(annotations):
    batch_size = 32
    batches = []

    for a_idx in range(len(annotations)//32+1):
        batch_bgn = a_idx*32
        batch_end = min(len(annotations), (a_idx+1)*32)
        img_names = []
        for annotation in annotations[batch_bgn:batch_end]:
            img_names.append("VOCdevkit/VOC2007/JPEGImages/" + annotation["filename"])
        batch = predict_batch(img_names, (224, 224), headless_conv)
        batches.append(batch)
        print("batch " +str(a_idx) + " prepared") 
    return np.vstack(batches)

# warning this may take some time!
reprs = compute_representations(annotations)

# Serialize representations
h5f = h5py.File('voc_representaions.h5', 'w')
h5f.create_dataset('reprs', data=reprs)
h5f.close()
