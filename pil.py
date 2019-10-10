from PIL import Image
from PIL import ImageFilter
import os
import numpy as np

'''
save()
crop(x1, x2, y1, y2)
copy()
paste(imageOb, (x, y))
resize(w, h)
rotate(degree, expand=False)
transpose(Image.FLIP_LEFT_RIGHT)
getpixel((x, y))
putpixel(x,y), (r,g,b))
split() -> splits into individual bands. R,G,B
merge(image.mode, sourceimage) -> build a new multiband image
getcolors(maxcolors=256) -> returns an unsorted list of (count,pixel) values
list(image.getdata(band)) -> indicate what band to return (default is all bands). returns a sequence-like object
getextrema() -> 2-tuple containing min and max pixel value for each band
histogram(mask=None, extrema=None) -> returns a list containing pixel counts
    multiband, all bands are concatenated (so an RGB histogram has 768 values)
show()

'''

'''
for filename in os.listdir('.'):
    if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
        continue
    im = Image.open(filename)
    *operations*
'''

def tester(directory = './best-artworks-of-all-time/resized'):
    l = []
    for filename in os.listdir(directory):
        if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
            continue
        # im = Image.open(directory + '/' + filename)
        l.append(filename)
    print("Done")
    return l

def img(filename, directory = './best-artworks-of-all-time/resized'):
    return Image.open(directory + '/' + filename)

def grayscaleHisto(im):
    return im.convert('L').histogram()

def normalizeHisto(histo):
    totalPix = sum(histo)
    return [x/totalPix for x in histo]

def meanAndvar(histo, display=True):
    mean = sum(histo) / len(histo)
    t = []
    for x in histo:
        t.append((x - mean)**2)
    var = sum(t) / (len(t) - 1)
    if display:
        print("Mean:", mean, "Variance:",var)
    return (mean,var)

def detectEdges(im):
    edgeImg = im.filter(ImageFilter.FIND_EDGES)
    return edgeImg

def subtraction(im1, im2):
    buff1 = np.asarray(im1)
    buff2 = np.asarray(im2)
    buff3 = buff1 - buff2
    diffIm = Image.fromarray(buff3)
    return diffIm

def channelMeans(im, normalize = False, alterOriginal = False):
    ''' Returns channel means/variance (including a grayscale channel).
    if "normalize", then returns (means/variance, normalizedBands).
    If normalized, the individual bands can be returned, or the
    original image altered and returned.'''
    split = Image.Image.split(im)
    bands = [band.histogram() for band in split]
    gray = im.convert('L').histogram()
    d = []
    for i in range(len(bands)):
        d.append(meanAndvar(normalizeHisto(bands[i])))
    d.append(gray)
    if not normalize:
        return d
    for i in range(len(bands)):
        for j in range(len(bands[i])):
            bands[i][j] = (bands[i][j] - d[i][0]) / d[i][1]
    if not alterOriginal:
        return (d, bands)
    
            

def normalizePixels(im):
    pixels = list(im.getdata())
    N = len(pixels)
    if type(pixels[0]) == list:
        total = [0] * len(pixels[0])
        for pixel in pixels:
            for i in range(len(pixel)):
                total[i] += pixel[i]
        for x in range(im.width):
            for y in range(im.height):
                newpix = []
                oldpix = im.getpixel((x,y))
                for i in range(len(oldpix)):
                    newpix.append(oldpix[i] - (total[i] / N))
                im.putpixel((x,y),tuple(newpix))
                    
                
    else:
        total = 0
        for p in pixels:
            total += p
        mean = total / len(pixels)
