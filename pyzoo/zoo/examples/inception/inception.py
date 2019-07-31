from optparse import OptionParser
from math import ceil
from datetime import datetime

from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.nnframes import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.models import inception
from zoo.pipeline.estimator import *


def config_option_parser():
    parser = OptionParser()
    parser.add_option("-f", "--folder", type=str, dest="folder", default="",
                      help="url of hdf+s folder store the hadoop sequence files")
    parser.add_option("--model", type=str, dest="model", default="", help="model snapshot location")
    parser.add_option("--state", type=str, dest="state", default="", help="state snapshot location")
    parser.add_option("--checkpoint", type=str, dest="checkpoint", default="", help="where to cache the model")
    parser.add_option("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                      help="overwrite checkpoint files")
    parser.add_option("-e", "--maxEpoch", type=int, dest="maxEpoch", default=0, help="epoch numbers")
    parser.add_option("-i", "--maxIteration", type=int, dest="maxIteration", default=3100, help="iteration numbers")
    parser.add_option("-l", "--learningRate", type=float, dest="learningRate", default=0.01, help="learning rate")
    parser.add_option("--warmupEpoch", type=int, dest="warmupEpoch", default=0, help="warm up epoch numbers")
    parser.add_option("--maxLr", type=float, dest="maxLr", default=0.0, help="max Lr after warm up")
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", help="batch size")
    parser.add_option("--classNum", type=int, dest="classNum", default=1000, help="class number")
    parser.add_option("--weightDecay", type=float, dest="weightDecay", default=0.0001, help="weight decay")
    parser.add_option("--checkpointIteration", type=int, dest="checkpointIteration", default=620,
                      help="checkpoint interval of iterations")
    parser.add_option("--gradientMin", type=float, dest="gradientMin", default=0.0, help="min gradient clipping by")
    parser.add_option("--gradientMax", type=float, dest="gradientMax", default=0.0, help="max gradient clipping by")
    parser.add_option("--gradientL2NormThreshold", type=float, dest="gradientL2NormThreshold", default=0.0,
                      help="gradient L2-Norm threshold")
    parser.add_option("--memoryType", type=str, dest="memoryType", default="DRAM", help="memory storage type, DRAM or PMEM")

    return parser


def get_inception_data(url, sc=None, data_type="train"):
    path = os.path.join(url, data_type)
    return SeqFileFolder.files_to_image_frame(url=path, sc=sc, class_num=1000)


if __name__ == "__main__":
    # parse options
    parser = config_option_parser()
    (options, args) = parser.parse_args(sys.argv)

    if not options.learningRate:
        parser.error("-l --learningRate is a mandatory opt")
    if not options.batchSize:
        parser.error("-b --batchSize is a mandatory opt")

    # init
    sc = init_nncontext("inception v1")

    image_size = 224  # create dataset
    train_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
                                              ImageRandomCrop(image_size, image_size),
                                              ImageRandomPreprocessing(ImageHFlip(), 0.5),
                                              ImageChannelNormalize(123.0, 117.0, 104.0),
                                              ImageMatToTensor(format="NHWC", to_RGB=True),
                                              ImageSetToSample(input_keys=["imageTensor"], target_keys=["label"])
                                              ])

    raw_train_data = get_inception_data(options.folder, sc, "train")
    train_data = FeatureSet.image_frame(raw_train_data).transform(train_transformer)

    val_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
                                            ImageRandomCrop(image_size, image_size),
                                            ImageRandomPreprocessing(ImageHFlip(), 0.5),
                                            ImageChannelNormalize(123.0, 117.0, 104.0),
                                            ImageMatToTensor(format="NHWC", to_RGB=True),
                                            ImageSetToSample(input_keys=["imageTensor"], target_keys=["label"])
                                            ])
    raw_val_data = get_inception_data(options.folder, sc, "val")
    val_data = FeatureSet.image_frame(raw_val_data).transform(val_transformer)

    # build model
    if options.model != "":
        # load model snapshot
        inception_model = Model.load(options.model)
    else:
        inception_model = inception.inception_v1_no_aux_classifier(options.classNum)

    # set optimization method
    iterationPerEpoch = int(ceil(float(1281167) / options.batchSize))
    if options.maxEpoch:
        maxIteration = iterationPerEpoch * options.maxEpoch
    else:
        maxIteration = options.maxIteration
    warmup_iteration = options.warmupEpoch * iterationPerEpoch
    if options.state != "":
        # load state snapshot
        optim = OptimMethod.load(options.state)
    else:
        if warmup_iteration == 0:
            warmupDelta = 0.0
        else:
            if options.maxLr:
                maxlr = options.maxLr
            else:
                maxlr = options.learningRate
            warmupDelta = (maxlr - options.learningRate)/warmup_iteration
        polyIteration = maxIteration - warmup_iteration
        lrSchedule = SequentialSchedule(iterationPerEpoch)
        lrSchedule.add(Warmup(warmupDelta), warmup_iteration)
        lrSchedule.add(Poly(0.5, maxIteration), polyIteration)
        optim = SGD(learningrate=options.learningRate, learningrate_decay=0.0, weightdecay=options.weightDecay,
                    momentum=0.9, dampening=0.0, nesterov=False,
                    leaningrate_schedule=lrSchedule)

    # create triggers
    if options.maxEpoch:
        checkpoint_trigger = EveryEpoch()
        test_trigger = EveryEpoch()
        end_trigger = MaxEpoch(options.maxEpoch)
    else:
        checkpoint_trigger = SeveralIteration(options.checkpointIteration)
        test_trigger = SeveralIteration(options.checkpointIteration)
        end_trigger = MaxIteration(options.maxIteration)

    timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Estimator
    estimator = Estimator(inception_model, optim_methods=optim, model_dir=options.checkpoint)

    if options.gradientMin and options.gradientMax:
        estimator.set_constant_gradient_clipping(options.gradientMin, options.gradientMax)

    if options.gradientL2NormThreshold:
        estimator.set_l2_norm_gradient_clipping(options.gradientL2NormThreshold)

    estimator.train_imagefeature(train_set=train_data,
                                 criterion=ClassNLLCriterion(),
                                 end_trigger=end_trigger,
                                 checkpoint_trigger=checkpoint_trigger,
                                 validation_set=val_data,
                                 validation_method=[Top1Accuracy(), Top5Accuracy()],
                                 batch_size=options.batchSize)

    inception_model.saveModel("/tmp/inception/model.bigdl", "/tmp/inception/model.bin", True)

    sc.stop()