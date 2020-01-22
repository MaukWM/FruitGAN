'''
Author:
Ruud van den Berg, s1381059
r.vandenberg-4@student.utwente.nl

Description:
Reads images.
Then it makes them square by padding white.
Then it scales them to a specified size.
At last the images are rotated and flipped.

History:
- v0.1, 2020-01-09
	Initial release

- v1.0, 2020-01-10
	Data augmentation added by rotating and mirroring images.

- v1.1, 2020-01-14
	The source directory and destination directory are specified at the top of the file.
	A flag can be set to specify whether the images have to be augmented or not.

- v1.2, 2020-01-16
	The initial numbering of the output images can be specified so that existing images
	are not overwritten.
'''

from PIL import Image, ImageOps
import os

fill_color = (255, 255, 255)
outputSize = 48
srcDir = "oranges"
destDir = "processed"
startNr = 0
augment = False

# 1. make list of image names
im_names = os.listdir(srcDir)

# 2. for all images in the list
for i in range(len(im_names)):
    # read images
    im = Image.open(srcDir + '/' + im_names[i])

    # make images square
    x, y = im.size
    size = max(x, y)
    im_square = Image.new('RGB', (size, size), fill_color)
    im_square.paste(im, (int((size - x) / 2), int((size - y) / 2)))

    # scale image
    im_scaled = im_square.resize((outputSize, outputSize))

    # rotate images
    im_mirrored = ImageOps.mirror(im_scaled)
    im_flipped = ImageOps.flip(im_scaled)

    # flip images
    im_090 = im_scaled.rotate(90)
    im_090_mir = ImageOps.mirror(im_090)
    im_180 = im_scaled.rotate(180)
    im_270 = im_scaled.rotate(270)
    im_270_mir = ImageOps.mirror(im_270)

    # store images
    im_scaled.save(destDir + '/' + str(i + startNr) + '_original.png')
    if augment:
        im_mirrored.save(destDir + '/' + str(i + startNr) + '_scaled.png')
        im_flipped.save(destDir + '/' + str(i + startNr) + '_flipped.png')
        im_090.save(destDir + '/' + str(i + startNr) + '_rot090.png')
        im_090_mir.save(destDir + '/' + str(i + startNr) + '_rot090_mir.png')
        im_180.save(destDir + '/' + str(i + startNr) + '_rot180.png')
        im_270.save(destDir + '/' + str(i + startNr) + '_rot270.png')
        im_270_mir.save(destDir + '/' + str(i + startNr) + '_rot270_mir.png')