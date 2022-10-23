import os
import glob
import numpy as np

# Author: Subhransu Maji (smaji@cs.umass.edu)
# Aug 3, 2022 @ Delft, NL

# Load the flickr material dataset 
def get_fmd(dataDir):
    imageDir = os.path.join(dataDir, 'image')
    classNames = [f for f in os.listdir(imageDir) if not f.startswith('.')]
    imageNames = glob.glob(os.path.join(imageDir, '*', '*.jpg'))
    numImages = len(imageNames)
    numClasses = len(classNames)

    # Initialize data structures
    meta, image, imdb = {},{},{}
    meta['dataset'] = 'FMD'
    meta['imageDir'] = imageDir
    meta['class'] = classNames

    # Mapping from classNames to ID
    classToId = {}
    for i in range(numClasses):
        classToId[classNames[i]] = i

    imageId = np.arange(0, numImages)
    classId = np.ones((numImages,1), np.int32)*-1;
    
    image['id'] = imageId
    image['name'] = imageNames

    for i in range(numImages):
        thisImageName = image['name'][i]
        xx = thisImageName.split('/')
        thisImageClass = xx[-2]
        classId[i] = classToId[thisImageClass]
    image['classId'] = classId

    imdb['meta'] = meta
    imdb['image'] = image

    print('+ dataset has {} images, {} classes.'.format(numImages,numClasses))
    return imdb;
    
# Load the describable texture dataset
def get_dtd(dataDir):
    imageDir = os.path.join(dataDir, 'images')
    labelDir = os.path.join(dataDir, 'labels')
    classNames = [f for f in os.listdir(imageDir) if not f.startswith('.')]
    
    with open(os.path.join(labelDir,'test1.txt')) as f:
        imageNames_ = f.readlines()

    numImages = len(imageNames_)
    numClasses = len(classNames)

    # Initialize data structures
    meta, image, imdb = {},{},{} 

    meta['dataset'] = 'FMD'
    meta['imageDir'] = imageDir
    meta['class'] = classNames

    # Mapping from classNames to ID
    classToId = {}
    for i in range(numClasses):
        classToId[classNames[i]] = i

    imageId = np.arange(0, numImages)
    classId = np.ones((numImages,1), np.int32)*-1;
    
    image['id'] = imageId
    imageNames = [];
    for i in range(numImages):
        thisImageName = imageNames_[i].rstrip('\n')
        imageNames.append(os.path.join(imageDir, thisImageName))
        xx = thisImageName.split('/')
        thisImageClass = xx[-2]
        classId[i] = classToId[thisImageClass]
    image['classId'] = classId
    image['name'] = imageNames

    imdb['meta'] = meta
    imdb['image'] = image

    print('+ dataset has {} images, {} classes.'.format(numImages,numClasses))
    return imdb;

# Return the imdb data scructure for KTH-TIPS
def get_kth(dataDir):
    imageDir = dataDir
    classNames = [f for f in os.listdir(imageDir) if os.path.isdir(os.path.join(imageDir, f))]
    imageNames = glob.glob(os.path.join(imageDir,'*','*.png'))
    numImages = len(imageNames)
    numClasses = len(classNames)

    # Initialize data structures
    meta, image, imdb = {},{},{} 

    meta['dataset'] = 'KTH-TIPS'
    meta['imageDir'] = imageDir
    meta['class'] = classNames

    # Mapping from classNames to ID
    classToId = {}
    for i in range(numClasses):
        classToId[classNames[i]] = i

    imageId = np.arange(0, numImages)
    classId = np.ones((numImages,1), np.int32)*-1;
    
    image['id'] = imageId
    image['name'] = imageNames

    for i in range(numImages):
        thisImageName = image['name'][i]
        xx = thisImageName.split('/')
        thisImageClass = xx[-2]
        classId[i] = classToId[thisImageClass]
    image['classId'] = classId

    imdb['meta'] = meta
    imdb['image'] = image

    print('+ dataset has {} images, {} classes.'.format(numImages,numClasses))
    return imdb;

# Return the imdb data scructure for KTH-TIPS
def get_kth2a(dataDir):
    imageDir = dataDir
    classNames = [f for f in os.listdir(imageDir) if os.path.isdir(os.path.join(imageDir, f))]
    imageNames = glob.glob(os.path.join(imageDir,'*','*','*.png'))
    numImages = len(imageNames)
    numClasses = len(classNames)

    # Initialize data structures
    meta, image, imdb = {},{},{} 

    meta['dataset'] = 'KTH-TIPS'
    meta['imageDir'] = imageDir
    meta['class'] = classNames

    # Mapping from classNames to ID
    classToId = {}
    for i in range(numClasses):
        classToId[classNames[i]] = i

    imageId = np.arange(0, numImages)
    classId = np.ones((numImages,1), np.int32)*-1;
    
    image['id'] = imageId
    image['name'] = imageNames

    for i in range(numImages):
        thisImageName = image['name'][i]
        xx = thisImageName.split('/')
        thisImageClass = xx[-3]
        classId[i] = classToId[thisImageClass]
    image['classId'] = classId

    imdb['meta'] = meta
    imdb['image'] = image

    print('+ dataset has {} images, {} classes.'.format(numImages,numClasses))
    return imdb;
