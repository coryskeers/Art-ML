#scikit-image
import numpy as np
import os
from skimage import data, io, img_as_float, filters

'''attributes
.shape
.size
indexable pixels

masking:
mask = image < #
image[mask] = pixel value

color image is array with an additional dimension for the channels.
(row, col, ch)'''

class Indexer:
    '''Creates an index for use in retrieving full paths for images in the given directory.'''
    def __init__(self, directory = "./best-artworks-of-all-time/resized/"):
        self.directory = directory
        self.index = self.buildIndex()
        
    def buildIndex(self):
        l = []
        try:
            for filename in os.listdir(self.directory):
                if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
                    continue
                l.append(filename)
            print("Index built")
            return l
        except:
            print("Error building index.")
            return -1

    def filepath(self, i):
        try:
            path = self.directory + self.index[i]
        except:
            print("Error retreiving path.")
            return -1
        return path
        

class Image:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = ''
        self.image = self.open(self.filepath)
        self.globalMean = -1
        self.localMean = -1
        self.globalVar = -1
        self.localVar = -1

    def open(self, filepath):
        if '/' in filepath:
            self.filename = filepath.split('/')[-1]
        else:
            self.filename = filepath
        try:
            im = io.imread(filepath)
            self.image = im
        except:
            self.image = -1
            print("Error opening file.")

    def isGrayscale(self):
        if type(self.image) == int:
            print("Image has not been opened.")
            return -1
        return len(self.image.shape) < 3

    def mean(self, local = False):
        if type(self.image) == int:
            print("Image has not been opened.")
        if not local:
            if self.globalMean == -1:
                self.globalMean = np.mean(self.image)
            return self.globalMean
        if self.localMean == -1:
            chmeans = []
            for ch in range(len(self.image[0][0])):
                chmeans.append(self.image[:,:,ch].mean())
            self.localMean = chmeans
        return self.localMean
    
    def var(self, local = False):
        if type(self.image) == int:
            print("Image has not been opened.")
        if not local:
            if self.globalVar == -1:
                if self.globalMean == -1:
                    self.mean(local = False)
                squares = np.sum((self.image - self.globalMean)**2)
                self.globalVar = squares / (self.image.shape[0] * self.image.shape[1])
            return self.globalVar
        if self.localVar == -1:
            chvars = []
            if self.localMean == -1:
                self.mean(local = True)
            for ch in range(len(self.image[0][0])):
                squares = np.sum((self.image[:,:,ch] - self.localMean[ch])**2)
                chvars.append(squares / (self.image.shape[0] * self.image.shape[1]))
            self.localVar = chvars
            return self.localVar

    def normalize(self, local = False, mean = -255, var = -255):
        if not local:
            if mean == -255:
                mean = self.globalMean
            if var == -255:
                var = self.globalVar
            self.normalized = (self.image - mean) / (var**(1/2))
        else:
            if type(mean) == int:
                mean = self.localMean
            if type(var) == int:
                var = self.localVar
            chnorms = []
            for ch in range(len(self.image[0][0])):
                chnorms.append((self.image[:,:,ch] - mean[ch]) / (var[ch]**(1/2)))
            self.normalized = numpy.array(chnorms)
        return self.normalized
