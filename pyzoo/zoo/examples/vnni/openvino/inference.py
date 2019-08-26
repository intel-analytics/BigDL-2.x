from optparse import OptionParser

from zoo.pipeline.inference import InferenceModel
from zoo.common.nncontext import init_nncontext
from zoo.feature.image import *
from zoo.pipeline.nnframes import *

batch_size = 4

def predict(model_path, img_path, partition_num):
    model = InferenceModel()
    model.load_openvino(model_path,
                        weight_path=model_path[:model_path.rindex(".")] + ".bin",
                        batch_size=batch_size)
    sc = init_nncontext("OpenVINO Object Detection Inference Example")
    infer_transformer = ChainedPreprocessing([ImageBytesToMat(),
                                             ImageResize(256, 256),
                                             ImageCenterCrop(224, 224),
                                             ImageMatToTensor(format="NHWC", to_RGB=True)])
    image_set = ImageSet.read(img_path, sc, partition_num).\
        transform(infer_transformer).get_image().collect()
    image_set = np.expand_dims(image_set, axis=1)

    if len(image_set) % batch_size == 0:
        a = 0
        size = batch_size
    else:
        a = 1
        size = len(image_set) % batch_size
    for i in range(len(image_set) // batch_size + a):
        index = i * batch_size
        batch = image_set[index]
        for j in range(index + 1, index + size):
            batch = np.vstack((batch, image_set[j]))
        batch = np.expand_dims(batch, axis=0)

        predictions = model.predict(batch)

        result = predictions[0]

        print("batch_" + str(i))
        for r in result:
            output = {}
            max_index = np.argmax(r)
            output["Top-1"] = str(max_index)
            print("* Predict result " + str(output))


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--image", type=str, dest="img_path",
                      help="The path where the images are stored, "
                           "can be either a folder or an image path")
    parser.add_option("--model", type=str, dest="model_path",
                      help="Zoo Model Path")
    parser.add_option("--partition_num", type=int, dest="partition_num", default=4,
                      help="The number of partitions")

    (options, args) = parser.parse_args(sys.argv)

    predict(options.model_path, options.img_path, options.partition_num)
