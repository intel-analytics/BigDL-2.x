from optparse import OptionParser

from zoo.pipeline.inference import InferenceModel
from zoo.common.nncontext import init_nncontext
from zoo.feature.image import *
from zoo.pipeline.nnframes import *

batch_size = 4


def predict(model_path, img_path):
    model = InferenceModel()
    model.load_openvino(model_path,
                        weight_path=model_path[:model_path.rindex(".")] + ".bin",
                        batch_size=batch_size)
    sc = init_nncontext("OpenVINO Python resnet_v1_50 Inference Example")
    # pre-processing
    infer_transformer = ChainedPreprocessing([ImageBytesToMat(),
                                             ImageResize(256, 256),
                                             ImageCenterCrop(224, 224),
                                             ImageMatToTensor(format="NHWC", to_RGB=True)])
    image_set = ImageSet.read(img_path, sc).\
        transform(infer_transformer).get_image().collect()
    image_set = np.expand_dims(image_set, axis=1)

    for i in range(len(image_set) // batch_size + 1):
        index = i * batch_size
        # check whether out of index
        if index >= len(image_set):
            break
        batch = image_set[index]
        # put 4 images in one batch
        for j in range(index + 1, min(index + batch_size, len(image_set))):
            batch = np.vstack((batch, image_set[j]))
        batch = np.expand_dims(batch, axis=0)
        # predict batch
        predictions = model.predict(batch)
        result = predictions[0]

        # post-processing for Top-1
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

    (options, args) = parser.parse_args(sys.argv)
    predict(options.model_path, options.img_path)
