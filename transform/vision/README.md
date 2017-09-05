# Vision

## Image Augmentation

Currently a series of image augmentation transformers have been supported:
* Brightness: adjust the image brightness
* ChannelOrder: shuffle the channel order
* Hue: adjust the image hue
* Saturation: adjust the image saturation
* Contrast: adjust image contrast
* ColorJitter: a random combination of **Brightness**, **ChannelOrder**, **Hue**, **Saturation** and **Contrast**.
* Resize: resize the image, default mode is INTER_LINEAR.
* HFlip: horizontally flip the image
* Expand: expand the image, with the background filled with given color
* Crop: crop the image given a roi(region of interest)
* CenterCrop: crop the center of image given width and height
* RandomCrop: crop the random area of image given width and height