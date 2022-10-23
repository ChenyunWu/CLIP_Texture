import torch
import CLIP.clip as clip
import os
import numpy as np
from PIL import Image
from dataloader import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Author: Subhransu Maji (smaji@cs.umass.edu)
# Aug 3, 2022 @ Delft, NL

# Experiment setup
dataSet = 'FMD' # 'DTD', 'FMD', 'KTH-TIPS', 'KTH-TIPS2a' 
modelName = "ViT-B/32"
expName = 'prompt-a-photo-of-a-{}-pattern-ViT-B32-' + dataSet
expDir = os.path.join('exp', expName)

print('Dataset ' + dataSet)
print('+ model ' + modelName)
print('+ expDir ' + expDir)
if not os.path.isdir(expDir):
    os.makedirs(expDir)
    print('+ expDir does not exist, creating one.')
    
# Get dataset
if dataSet == 'DTD':
    imdb = get_dtd(os.path.join('data', dataSet))
elif dataSet == 'FMD':
    imdb = get_fmd(os.path.join('data', dataSet))
elif dataSet == 'KTH-TIPS':
    imdb = get_kth(os.path.join('data', dataSet))
elif dataSet == 'KTH-TIPS2a':
    imdb = get_kth2a(os.path.join('data', dataSet))
else:
    print('Dataset not supported. Aborting!')

resultsFile = os.path.join(expDir, 'predId.npy')

if not os.path.isfile(resultsFile):
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(modelName, device=device)
    print('+ loaded model ' + modelName)

    # Prompts for zero-shot classification
    classPrompts = [f"a photo of a {c} pattern" for c in imdb['meta']['class']]
    print('+ prompts: ' + str(classPrompts))

    # Extract text features
    text_input = clip.tokenize(classPrompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Extract image features and match to text
    numCorrect = 0
    numImages = len(imdb['image']['name'])
    predId = np.ones((numImages,1), np.int32)*-1;
    for i in range(numImages):
        imageName = imdb['image']['name'][i]
        image_input = preprocess(Image.open(imageName)).unsqueeze(0).to(device);
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)
            
        predId[i] = indices[0].cpu()

        if i % 100 == 0:
            numCorrect = np.sum(predId == imdb['image']['classId'])
            print('+ processed {} of {} images, {} correct'.format(i, numImages, numCorrect))

    print("[done]")
    np.save(resultsFile, predId)
    print('+ saved predictions.')
else:
    predId = np.load(resultsFile)
    print('+ loaded precomputed predictions.')

# Print results and save figures
numCorrect = np.sum(predId == imdb['image']['classId'])
accuracy = numCorrect*100.0 / len(predId)
print('Total correct = {}/{} (Accuracy {:.2f}%)'.format(numCorrect, len(predId), accuracy))

conf = confusion_matrix(imdb['image']['classId'],predId)
print(conf)
disp = ConfusionMatrixDisplay(confusion_matrix=conf,
                              display_labels = imdb['meta']['class'])
disp.plot(xticks_rotation=45)
plt.title('{} {} Accuracy {:.2f} %'.format(dataSet, modelName, accuracy))
plt.savefig(os.path.join(expDir, 'conf.pdf'))
