from bigdl.util.common import *
from scipy import misc
from image import *
from pipeline import *
import cv2
import matplotlib.pyplot as plt
import os.path
from bigdl.nn.layer import Model

def init():
    JavaCreator.set_creator_class("com.intel.analytics.zoo.pipeline.common.pythonapi.PythonPipeline")
    init_engine()
    
def load_local_folder(folder):
    imageFiles = os.listdir(folder)
    images = []
    for f in imageFiles:
        image = cv2.imread(folder + '/' + f)
        images.append(image)
    return images

def load_pascal_classes(filename=None):
    class_file = filename if filename else "../../pipeline/ssd/data/pascal/classname.txt"
    assert os.path.isfile(class_file), 'pascal class file ' + class_file + ' does not exists!'
    with open(class_file) as f:
        return f.read().splitlines()
        
def show_images(images):
    for image in images:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

# data preprocess
def preprocess_ssd(img_rdd, means_rgb, scale, resolution, n_partition, batch_per_partition):
    sizes = img_rdd.map(lambda img: img.shape)
    transformer = Pipeline([Resize(resolution, resolution),
                            ChannelNormalize(means_rgb[0], means_rgb[1], means_rgb[2], scale, scale, scale),
                            MatToFloats(resolution, resolution)])
    # create ImageFrame from image ndarray rdd
    image_frame = ImageFrame(img_rdd)
    image_frame = transformer(image_frame)
    return to_ssd_batch(image_frame, n_partition, batch_per_partition)

def preprocess_ssd_mobilenet(img_rdd, resolution = 300, n_partition=4, batch_per_partition=1):
    means = np.array([127.5, 127.5, 127.5]) # mean value in RGB order
    scale = 1 / 0.007843
    return preprocess_ssd(img_rdd, means, scale, resolution, n_partition, batch_per_partition)

def preprocess_ssd_vgg(img_rdd, resolution = 300, n_partition=4, batch_per_partition=1):
    means = np.array([123.0, 117.0, 104.0]) # mean value in RGB order
    scale = 1.0
    return preprocess_ssd(img_rdd, means, scale, resolution, n_partition, batch_per_partition)

def preprocess_frcnn(img_rdd, resolution=600, scale_multiple_of=1, n_partition=4):
    means_rgb = np.array([122.7717, 115.9465, 102.9801])
    scale = 1.0
    transformer = Pipeline([AspectScale(resolution, scale_multiple_of),
                            ChannelNormalize(means_rgb[0], means_rgb[1], means_rgb[2], scale, scale, scale),
                            MatToFloats(resolution, resolution)])
    # create ImageFrame from image ndarray rdd
    image_frame = ImageFrame(img_rdd)
    image_frame = transformer(image_frame)
    return to_frcnn_batch(image_frame, n_partition)

def preprocess_frcnn_pvanet(img_rdd, resolution=640, n_partition=4):
    return preprocess_frcnn(img_rdd, resolution, 32, n_partition)
    
def preprocess_frcnn_vgg(img_rdd, resolution=600, n_partition=4):
    return preprocess_frcnn(img_rdd, resolution, 1, n_partition)

# Visualize detection to original images
def visualize(img, detections, classes, threshold = 0.6):
    total = detections.shape[0]
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for i in range(0, total):
        score = detections[i][1]
        cls_id = int(detections[i][0])
        if (score <= threshold):
            continue
        cv2.rectangle(img,(detections[i][2],detections[i][3]),(detections[i][4],detections[i][5]),(0,255,0),3)
        cv2.putText(img,'{:s} {:.3f}'.format(classes[cls_id], score),
                    (int(detections[i][2]),int(detections[i][3] - 2)), font, 1,(255,255,255),1,cv2.LINE_AA)
    return img

def visualize_detections(images, result, classes):
    # visualize detections
    for img_id in range(len(result)):
        detections = result[img_id]
        visualize(images[img_id], detections, classes)
