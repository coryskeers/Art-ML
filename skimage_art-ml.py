#scikit-image
import numpy as np
import os
import matplotlib.pyplot as plt

from skimage import data, io, img_as_float, filters
from skimage.color import rgb2gray

'''attributes
.shape
.size
indexable pixels

masking:
mask = image < #
image[mask] = pixel value

color image is array with an additional dimension for the channels.
(row, col, ch)'''

class Runner:
    def __init__(self):
        self.ind = Indexer()

    def analyzeArtist(self, artist):
        self.artist = artist
        self.artistData = []
        index = []
        for f in self.ind.index:
            if artist in f.lower():
                index.append(f)
        for filename in index:
            d = [filename]
            im = Image(self.ind.directory + filename)
            d.append(im.mean())
            lm = im.mean(True)
            for i in range(3):
                if type(lm) == list and len(lm) - 1 >= i:
                    d.append(lm[i])
                else:
                    d.append(0)
            d.append(im.var())
            lv = im.var(True)
            for i in range(3):
                if type(lv) == list and len(lv) - 1 >= i:
                    d.append(lv[i])
                else:
                    d.append(0)            
            d.append(im.edgeDetect(True))
            d.append(im.invert(True))
            self.artistData.append(d)
        self.write(artist + '.txt', self.artistData)

    def analyzeTitian(self):
        self.titianData = []
        for filename in self.titian:
            d = [filename]
            im = Image(self.ind.directory + filename)
            d.append(im.mean())
            d.append(im.var())
            d.append(im.edgeDetect(True))
            d.append(im.invert(True))
            self.titianData.append(d)

    def write(self, filename, data):
        with open(filename, 'w') as file:
            for line in data:
                for d in range(len(line)):
                    file.write(str(line[d]))
                    file.write(",")
                file.write("\n")

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
        self.open()
        self.globalMean = -1
        self.localMean = -1
        self.globalVar = -1
        self.localVar = -1
        self.normalized = -1
        self.edges = -1
        self.inverted = -1
        self.hflip = -1
        self.grayHistogram = -1
        self.histogram = -1

    def open(self):
        if '/' in self.filepath:
            self.filename = self.filepath.split('/')[-1]
        else:
            self.filename = self.filepath
        try:
            im = io.imread(self.filepath)
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
        if self.localMean == -1 and len(self.image.shape) > 2:
            chmeans = []
            for ch in range(self.image.shape[2]):
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
                self.globalVar = squares / ((self.image.shape[0] * self.image.shape[1]) - 1)
            return self.globalVar
        if self.localVar == -1 and len(self.image.shape) > 2:
            chvars = []
            if self.localMean == -1:
                self.mean(local = True)
            for ch in range(self.image.shape[2]):
                squares = np.sum((self.image[:,:,ch] - self.localMean[ch])**2)
                chvars.append(squares / ((self.image.shape[0] * self.image.shape[1])) - 1)
            self.localVar = chvars
            return self.localVar
    
    def histograms(self, gray = False):
        if type(self.image) == int:
            print("Image has not been opened.")
        if gray:
            if type(self.grayHistogram) == int:
                pixels = rgb2gray(self.image).flatten()
                self.grayHistogram = [0] * 256;
                for pixel in pixels:
                    self.grayHistogram[pixel] += 1
            return self.grayHistogram
        if type(self.histogram) == int:
            chhistos = []
            for ch in range(len(self.image[0][0])):
                histo = [0] * 256
                pixels = self.image[:,:,ch].flatten()
                for pixel in pixels:
                    histo[pixel] += 1
                chhistos.append(histo)
            self.histogram = chhistos
        return self.histogram

    def normalize(self, local = False, mean = -255, var = -255):
        if type(self.normalized) != int:
            return self.normalized
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
            for ch in range(self.image.shape[2]):
                chnorms.append((self.image[:,:,ch] - mean[ch]) / (var[ch]**(1/2)))
            self.normalized = np.array(chnorms)
        return self.normalized

    def edgeDetect(self, sharpIndex = False):
        if type(self.edges) == int:
            self.edges = filters.sobel(rgb2gray(self.image))
        if sharpIndex:
            # Higher index suggests higher rate of sharp edges
            return np.mean(self.edges)
        return self.edges

    def invert(self, symmetryIndex = False):
        if type(self.inverted) == int:
            self.inverted = 255 - self.image
        if symmetryIndex:
            # Higher index suggests higher rate of horizontal symmetry
            # 1.0 would indicate perfect symmetry
            return (np.mean(rgb2gray(self.inverted) + rgb2gray(self.flip())) + 1)/256
        return self.inverted

    def flip(self):
        if type(self.hflip) == int:
            self.hflip = np.copy(self.image)
            for i in range(self.image.shape[1]):
                if (len(self.image.shape) > 2):
                    self.hflip[:,i,:] = self.image[:,-i - 1,:]
                else:
                    self.hflip[:,i] = self.image[:,-i - 1] 
        return self.hflip
    
    def show(self, image = -255):
        if type(image) == int:
            image = self.image
        io.imshow(image)
        io.show()
