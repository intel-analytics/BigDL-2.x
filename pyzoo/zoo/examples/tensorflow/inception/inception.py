#from bigdl.nn.layer import Sequential, Concat, SpatialConvolution, ReLU, SpatialMaxPooling, SpatialCrossMapLRN, SpatialAveragePooling, Dropout, View, Linear, LogSoftMax
#from optparse import OptionParser
#from bigdl.nn.criterion import *
#from bigdl.nn.initialization_method import *
#from bigdl.optim.optimizer import *
#from zoo.feature.image.imagePreprocessing import *
#from zoo.feature.image.imageset import *
#from zoo.feature.common import ChainedPreprocessing
#from math import ceil
#from zoo.common.nncontext import *
#from zoo.feature.common import *
#from zoo.feature.image.imagePreprocessing import *
#
#from zoo.common.nncontext import *
#from zoo.feature.common import *
#from zoo.feature.image.imagePreprocessing import *
#from zoo.pipeline.api.keras.layers import Dense, Input, Flatten
#from zoo.pipeline.api.net import *
#from zoo.pipeline.api.keras.models import Model
#from bigdl.optim.optimizer import *
#
#
#def t(input_t):
#    if type(input_t) is list:
#        # insert into index 0 spot, such that the real data starts from index 1
#        temp = [0]
#        temp.extend(input_t)
#        return dict(enumerate(temp))
#    # if dictionary, return it back
#    return input_t
#
#
#def inception_layer_v1(input_size, config, name_prefix=""):
#    concat = Concat(2)
#    conv1 = Sequential()
#    conv1.add(SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1)
#              .set_init_method(weight_init_method=Xavier(),bias_init_method=Zeros())
#              .set_name(name_prefix + "1x1"))
#    conv1.add(ReLU(True).set_name(name_prefix + "relu_1x1"))
#    concat.add(conv1)
#    conv3 = Sequential()
#    conv3.add(SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1)
#              .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#              .set_name(name_prefix + "3x3_reduce"))
#    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3_reduce"))
#    conv3.add(SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1)
#              .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#              .set_name(name_prefix + "3x3"))
#    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3"))
#    concat.add(conv3)
#    conv5 = Sequential()
#    conv5.add(SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1)
#              .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#              .set_name(name_prefix + "5x5_reduce"))
#    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5_reduce"))
#    conv5.add(SpatialConvolution(config[3][1], config[3][2], 5, 5, 1, 1, 2, 2)
#              .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#              .set_name(name_prefix + "5x5"))
#    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5"))
#    concat.add(conv5)
#    pool = Sequential()
#    pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1, to_ceil=True).set_name(name_prefix + "pool"))
#    pool.add(SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1)
#             .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#             .set_name(name_prefix + "pool_proj"))
#    pool.add(ReLU(True).set_name(name_prefix + "relu_pool_proj"))
#    concat.add(pool).set_name(name_prefix + "output")
#    return concat
#
#
#def inception_v1_no_aux_classifier(class_num, has_dropout=True):
#    model = Sequential()
#    model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, False)
#              .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#              .set_name("conv1/7x7_s2"))
#    model.add(ReLU(True).set_name("conv1/relu_7x7"))
#    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool1/3x3_s2"))
#    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("pool1/norm1"))
#    model.add(SpatialConvolution(64, 64, 1, 1, 1, 1)
#              .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#              .set_name("conv2/3x3_reduce"))
#    model.add(ReLU(True).set_name("conv2/relu_3x3_reduce"))
#    model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
#              .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#              .set_name("conv2/3x3"))
#    model.add(ReLU(True).set_name("conv2/relu_3x3"))
#    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("conv2/norm2"))
#    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool2/3x3_s2"))
#    model.add(inception_layer_v1(192, t([t([64]), t(
#        [96, 128]), t([16, 32]), t([32])]), "inception_3a/"))
#    model.add(inception_layer_v1(256, t([t([128]), t(
#        [128, 192]), t([32, 96]), t([64])]), "inception_3b/"))
#    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
#    model.add(inception_layer_v1(480, t([t([192]), t(
#        [96, 208]), t([16, 48]), t([64])]), "inception_4a/"))
#    model.add(inception_layer_v1(512, t([t([160]), t(
#        [112, 224]), t([24, 64]), t([64])]), "inception_4b/"))
#    model.add(inception_layer_v1(512, t([t([128]), t(
#        [128, 256]), t([24, 64]), t([64])]), "inception_4c/"))
#    model.add(inception_layer_v1(512, t([t([112]), t(
#        [144, 288]), t([32, 64]), t([64])]), "inception_4d/"))
#    model.add(inception_layer_v1(528, t([t([256]), t(
#        [160, 320]), t([32, 128]), t([128])]), "inception_4e/"))
#    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
#    model.add(inception_layer_v1(832, t([t([256]), t(
#        [160, 320]), t([32, 128]), t([128])]), "inception_5a/"))
#    model.add(inception_layer_v1(832, t([t([384]), t(
#        [192, 384]), t([48, 128]), t([128])]), "inception_5b/"))
#    model.add(SpatialAveragePooling(7, 7, 1, 1).set_name("pool5/7x7_s1"))
#    if has_dropout:
#        model.add(Dropout(0.4).set_name("pool5/drop_7x7_s1"))
#    model.add(View([1024], num_input_dims=3))
#    model.add(Linear(1024, class_num)
#              .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#              .set_name("loss3/classifier"))
#    model.add(LogSoftMax().set_name("loss3/loss3"))
#    model.reset()
#    return model
#
#
#def inception_v1(class_num, has_dropout=True):
#    feature1 = Sequential()
#    feature1.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, False)
#                 .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#                 .set_name("conv1/7x7_s2"))
#    feature1.add(ReLU(True).set_name("conv1/relu_7x7"))
#    feature1.add(
#        SpatialMaxPooling(3, 3, 2, 2, to_ceil=True)
#            .set_name("pool1/3x3_s2"))
#    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75)
#                 .set_name("pool1/norm1"))
#    feature1.add(SpatialConvolution(64, 64, 1, 1, 1, 1)
#                 .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#                 .set_name("conv2/3x3_reduce"))
#    feature1.add(ReLU(True).set_name("conv2/relu_3x3_reduce"))
#    feature1.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
#                 .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#                 .set_name("conv2/3x3"))
#    feature1.add(ReLU(True).set_name("conv2/relu_3x3"))
#    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("conv2/norm2"))
#    feature1.add(
#        SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool2/3x3_s2"))
#    feature1.add(inception_layer_v1(192, t([
#        t([64]), t([96, 128]), t([16, 32]), t([32])]),
#                                    "inception_3a/"))
#    feature1.add(inception_layer_v1(256, t([
#        t([128]), t([128, 192]), t([32, 96]), t([64])]),
#                                    "inception_3b/"))
#    feature1.add(
#        SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool3/3x3_s2"))
#    feature1.add(inception_layer_v1(480, t([
#        t([192]), t([96, 208]), t([16, 48]), t([64])]),
#                                    "inception_4a/"))
#
#    output1 = Sequential()
#    output1.add(SpatialAveragePooling(
#        5, 5, 3, 3, ceil_mode=True).set_name("loss1/ave_pool"))
#    output1.add(
#        SpatialConvolution(512, 128, 1, 1, 1, 1).set_name("loss1/conv"))
#    output1.add(ReLU(True).set_name("loss1/relu_conv"))
#    output1.add(View([128 * 4 * 4, 3]))
#    output1.add(Linear(128 * 4 * 4, 1024).set_name("loss1/fc"))
#    output1.add(ReLU(True).set_name("loss1/relu_fc"))
#    if has_dropout:
#        output1.add(Dropout(0.7).set_name("loss1/drop_fc"))
#    output1.add(Linear(1024, class_num).set_name("loss1/classifier"))
#    output1.add(LogSoftMax().set_name("loss1/loss"))
#
#    feature2 = Sequential()
#    feature2.add(inception_layer_v1(512,
#                                    t([t([160]), t([112, 224]), t([24, 64]), t([64])]),
#                                    "inception_4b/"))
#    feature2.add(inception_layer_v1(512,
#                                    t([t([128]), t([128, 256]), t([24, 64]), t([64])]),
#                                    "inception_4c/"))
#    feature2.add(inception_layer_v1(512,
#                                    t([t([112]), t([144, 288]), t([32, 64]), t([64])]),
#                                    "inception_4d/"))
#
#    output2 = Sequential()
#    output2.add(SpatialAveragePooling(5, 5, 3, 3).set_name("loss2/ave_pool"))
#    output2.add(
#        SpatialConvolution(528, 128, 1, 1, 1, 1).set_name("loss2/conv"))
#    output2.add(ReLU(True).set_name("loss2/relu_conv"))
#    output2.add(View([128 * 4 * 4, 3]))
#    output2.add(Linear(128 * 4 * 4, 1024).set_name("loss2/fc"))
#    output2.add(ReLU(True).set_name("loss2/relu_fc"))
#    if has_dropout:
#        output2.add(Dropout(0.7).set_name("loss2/drop_fc"))
#    output2.add(Linear(1024, class_num).set_name("loss2/classifier"))
#    output2.add(LogSoftMax().set_name("loss2/loss"))
#
#    output3 = Sequential()
#    output3.add(inception_layer_v1(528,
#                                   t([t([256]), t([160, 320]), t([32, 128]), t([128])]),
#                                   "inception_4e/"))
#    output3.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool4/3x3_s2"))
#    output3.add(inception_layer_v1(832,
#                                   t([t([256]), t([160, 320]), t([32, 128]), t([128])]),
#                                   "inception_5a/"))
#    output3.add(inception_layer_v1(832,
#                                   t([t([384]), t([192, 384]), t([48, 128]), t([128])]),
#                                   "inception_5b/"))
#    output3.add(SpatialAveragePooling(7, 7, 1, 1).set_name("pool5/7x7_s1"))
#    if has_dropout:
#        output3.add(Dropout(0.4).set_name("pool5/drop_7x7_s1"))
#    output3.add(View([1024, 3]))
#    output3.add(Linear(1024, class_num)
#                .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
#                .set_name("loss3/classifier"))
#    output3.add(LogSoftMax().set_name("loss3/loss3"))
#
#    split2 = Concat(2).set_name("split2")
#    split2.add(output3)
#    split2.add(output2)
#
#    mainBranch = Sequential()
#    mainBranch.add(feature2)
#    mainBranch.add(split2)
#
#    split1 = Concat(2).set_name("split1")
#    split1.add(mainBranch)
#    split1.add(output1)
#
#    model = Sequential()
#
#    model.add(feature1)
#    model.add(split1)
#
#    model.reset()
#    return model
#
#
#def get_inception_data(url, sc=None, data_type="train"):
#    path = os.path.join(url, data_type)
#    return SeqFileFolder.files_to_image_frame(url=path, sc=sc, class_num=1000)
#
#
#def config_option_parser():
#    parser = OptionParser()
#    parser.add_option("-f", "--folder", type=str, dest="folder", default="",
#                      help="url of hdfs folder store the hadoop sequence files")
#    parser.add_option("--model", type=str, dest="model", default="", help="model snapshot location")
#    parser.add_option("--state", type=str, dest="state", default="", help="state snapshot location")
#    parser.add_option("--checkpoint", type=str, dest="checkpoint", default="", help="where to cache the model")
#    parser.add_option("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
#                      help="overwrite checkpoint files")
#    parser.add_option("-e", "--maxEpoch", type=int, dest="maxEpoch", default=0, help="epoch numbers")
#    parser.add_option("-i", "--maxIteration", type=int, dest="maxIteration", default=62000, help="iteration numbers")
#    parser.add_option("-l", "--learningRate", type=float, dest="learningRate", default=0.01, help="learning rate")
#    parser.add_option("--warmupEpoch", type=int, dest="warmupEpoch", default=0, help="warm up epoch numbers")
#    parser.add_option("--maxLr", type=float, dest="maxLr", default=0.0, help="max Lr after warm up")
#    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", help="batch size")
#    parser.add_option("--classNum", type=int, dest="classNum", default=1000, help="class number")
#    parser.add_option("--weightDecay", type=float, dest="weightDecay", default=0.0001, help="weight decay")
#    parser.add_option("--checkpointIteration", type=int, dest="checkpointIteration", default=620,
#                      help="checkpoint interval of iterations")
#    parser.add_option("--gradientMin", type=float, dest="gradientMin", default=0.0, help="min gradient clipping by")
#    parser.add_option("--gradientMax", type=float, dest="gradientMax", default=0.0, help="max gradient clipping by")
#    parser.add_option("--gradientL2NormThreshold", type=float, dest="gradientL2NormThreshold", default=0.0, help="gradient L2-Norm threshold")
#
#    return parser
#
#
#if __name__ == "__main__":
#    # parse options
#    parser = config_option_parser()
#    (options, args) = parser.parse_args(sys.argv)
#    if not options.learningRate:
#        parser.error("-l --learningRate is a mandatory opt")
#    if not options.batchSize:
#        parser.error("-b --batchSize is a mandatory opt")
#
#    # init
#    sparkConf = create_spark_conf().setAppName("inception v1")
#    sc = get_spark_context(sparkConf)
#    redire_spark_logs()
#    show_bigdl_info_logs()
#    init_engine()
#
#    image_size = 224  # create dataset
#    train_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
#                                  ImageRandomCrop(image_size, image_size),
#                                  ImageRandomPreprocessing(ImageHFlip(), 0.5),
#                                  ImageChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
#                                  ImageMatToTensor(to_RGB=True),
#                                  ImageSetToSample(input_keys=["imageTensor"], target_keys=["label"])
#                                  ])
#    raw_train_data = get_inception_data(options.folder, sc, "train")
#    train_data = FeatureSet.image_frame(raw_train_data).transform(train_transformer).transform(ImageFeatureToSample())
#
#    val_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
#                                ImageCenterCrop(image_size, image_size),
#                                ImageRandomPreprocessing(ImageHFlip(), 0.5),
#                                ImageChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
#                                ImageMatToTensor(to_RGB=True),
#                                ImageSetToSample(input_keys=["imageTensor"], target_keys=["label"]),
#                                ])
#    raw_val_data = get_inception_data(options.folder, sc, "val")
#    val_data = FeatureSet.image_frame(raw_val_data).transform(val_transformer).transform(ImageFeatureToSample())
#
#    # build model
#    bigdl_model = inception_v1_no_aux_classifier(options.classNum)
#
#    model_path = "/tmp/bigdl_inception-v1_imagenet_0.4.0.model"
#    bigdl_model.saveModel(model_path, over_write = True)
#    inception_zoo_model = Net.load_bigdl(model_path)
#    inputNode = Input(name="input", shape=(3, 224, 224))
#    outputLayer = inception_zoo_model.to_keras()(inputNode)
#    model = Model(inputNode, outputLayer)
#
#
#    # set optimization method
#    iterationPerEpoch = int(ceil(float(1281167) / options.batchSize))
#    if options.maxEpoch:
#        maxIteration = iterationPerEpoch * options.maxEpoch
#    else:
#        maxIteration = options.maxIteration
#    warmup_iteration = options.warmupEpoch * iterationPerEpoch
#    if options.state != "":
#        # load state snapshot
#        optim = OptimMethod.load(options.state)
#    else:
#        if warmup_iteration == 0:
#            warmupDelta = 0.0
#        else:
#            if options.maxLr:
#                maxlr = options.maxLr
#            else:
#                maxlr = options.learningRate
#            warmupDelta = (maxlr - options.learningRate)/warmup_iteration
#        polyIteration = maxIteration - warmup_iteration
#        lrSchedule = SequentialSchedule(iterationPerEpoch)
#        lrSchedule.add(Warmup(warmupDelta), warmup_iteration)
#        lrSchedule.add(Poly(0.5, polyIteration), polyIteration)
#        optim = SGD(learningrate=options.learningRate, learningrate_decay=0.0, weightdecay=options.weightDecay,
#                    momentum=0.9, dampening=0.0, nesterov=False,
#                    leaningrate_schedule=lrSchedule)
#
#    # create triggers
#    if options.maxEpoch:
#        checkpoint_trigger = EveryEpoch()
#        test_trigger = EveryEpoch()
#        end_trigger = MaxEpoch(options.maxEpoch)
#    else:
#        checkpoint_trigger = SeveralIteration(options.checkpointIteration)
#        test_trigger = SeveralIteration(options.checkpointIteration)
#        end_trigger = MaxIteration(options.maxIteration)
#
#    # Optimizer
#    model.compile(optimizer=optim,
#                    loss='categorical_crossentropy',
#                    metrics=['accuracy', 'top5acc'])
#    model.fit(x = train_data, batch_size=options.batchSize, nb_epoch=70, validation_data=val_data)
#    # optimizer = Optimizer.create(
#    #     model=inception_model,
#    #     training_set=train_data,
#    #     optim_method=optim,
#    #     criterion=ClassNLLCriterion(),
#    #     end_trigger=end_trigger,
#    #     batch_size=options.batchSize
#    # )
#
#    # if options.checkpoint:
#    #     optimizer.set_checkpoint(checkpoint_trigger, options.checkpoint, options.overwrite)
#    #
#    # if options.gradientMin and options.gradientMax:
#    #     optimizer.set_gradclip_const(options.gradientMin, options.gradientMax)
#    #
#    # if options.gradientL2NormThreshold:
#    #     optimizer.set_gradclip_l2norm(options.gradientL2NormThreshold)
#
#    # optimizer.set_validation(trigger=test_trigger,
#    #                          val_rdd=val_data,
#    #                          batch_size=options.batchSize,
#    #                          val_method=[Top1Accuracy(), Top5Accuracy()])
#
#    # trained_model = optimizer.optimize()
#
#    sc.stop()

from optparse import OptionParser
from math import ceil

from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.api.net import *
from zoo.pipeline.nnframes import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

# import the inception model
import tensorflow as tf

#sys.path.append("/root/manfei/models/research/slim/")  # add the slim library
sys.path.append("/home/xin/IdeaProjects/models/research/slim")  # add the slim library
from nets import inception_v1

slim = tf.contrib.slim


def t(input_t):
    if type(input_t) is list:
        # insert into index 0 spot, such that the real data starts from index 1
        temp = [0]
        temp.extend(input_t)
        return dict(enumerate(temp))
    # if dictionary, return it back
    return input_t


def get_inception_data(url, sc=None, data_type="train"):
    path = os.path.join(url, data_type)
    return SeqFileFolder.files_to_image_frame(url=path, sc=sc, class_num=1000)


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

    return parser


if __name__ == "__main__":
    # parse options
    print("-----------set one option parser for user---------")
    parser = config_option_parser()
    print("-----------got the settings from the user---------")
    (options, args) = parser.parse_args(sys.argv)

    if not options.learningRate:
        parser.error("-l --learningRate is a mandatory opt")
    print("-----------check if the learning rate has been set---------")
    if not options.batchSize:
        parser.error("-b --batchSize is a mandatory opt")
    print("-----------check if the batchsize has been set---------")

    ## init
    sc = init_nncontext("inception v1")
    print("-----------init the spark???---------")

    image_size = 224  # create dataset
    print("-----------set the image size for this test, so we know that the image's size is 224---------")
    ## got dataFrame
    # transformer = ChainedPreprocessing(
    #     [RowToImageFeature(), ImageResize(256, 256), ImageCenterCrop(224, 224),
    #      ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(), ImageFeatureToTensor()])
    train_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
                                  ImageRandomCrop(image_size, image_size),
                                  ImageRandomPreprocessing(ImageHFlip(), 0.5),
                                  ImageChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                  ImageMatToTensor(format="NHWC", to_RGB=True),
                                  ImageSetToSample(input_keys=["imageTensor"], target_keys=["label"])
                                  ])
    raw_train_data = get_inception_data(options.folder, sc, "train")
    train_data = FeatureSet.image_frame(raw_train_data).transform(train_transformer)

    val_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
                                ImageCenterCrop(image_size, image_size),
                                ImageRandomPreprocessing(ImageHFlip(), 0.5),
                                ImageChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                ImageMatToTensor(format="NHWC", to_RGB=True),
                                ImageSetToSample(input_keys=["imageTensor"], target_keys=["label"]),
                                ])
    raw_val_data = get_inception_data(options.folder, sc, "val")
    val_data = FeatureSet.image_frame(raw_val_data).transform(val_transformer)

    #train_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
    #                                          ImageRandomCrop(image_size, image_size),
    #                                          ImageRandomPreprocessing(ImageHFlip(), 0.5),
    #                                          ImageChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
    #                                          ImageMatToTensor(to_RGB=True),
    #                                          ImageSetToSample(input_keys=["imageTensor"], target_keys=["label"])
    #                                          ])
    #raw_train_data = get_inception_data(options.folder, sc, "train")
    #train_data = ImageSet.from_image_frame(raw_train_data).transform(train_transformer)
    #print("-----------get the train data---------")

    #val_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
    #                                        ImageCenterCrop(image_size, image_size),
    #                                        ImageRandomPreprocessing(ImageHFlip(), 0.5),
    #                                        ImageChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
    #                                        ImageMatToTensor(to_RGB=True),
    #                                        ImageSetToSample(input_keys=["imageTensor"], target_keys=["label"])
    #                                        ])
    #raw_val_data = get_inception_data(options.folder, sc, "val")
    #val_data = ImageSet.from_image_frame(raw_val_data).transform(val_transformer)
    print("-----------get the validation data---------")

    dataset = TFDataset.from_feature_set(train_data,
                                         features=(tf.float32, [224, 224, 3]),
                                         labels=(tf.int32, [1]),
                                         batch_size=options.batchSize,
                                         validation_dataset=val_data)

    # construct the model from TFDataset
    images, labels = dataset.tensors

    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        logits, end_points = inception_v1.inception_v1(images, num_classes=1000, is_training=True)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

    # model_path = "/root/manfei/trainedmodels/bigdl_inception-v1_imagenet_0.1.0.model"
    # # bigdl_model.saveModel(model_path, over_write = True)
    # inception_zoo_model = Net.load_bigdl(model_path)
    # print("-----------loaded the model---------")
    # # print("*****get type*****")
    # # print(type(inception_zoo_model))
    #
    # inputNode = Input(name="input", shape=(3, 224, 224))
    # outputLayer = inception_zoo_model(inputNode)# GraphNet tranfor to keras or we can directlly get the output node using the input
    # #get loss form outputlayer and labels
    # model = Model(inputNode, outputLayer)## whether it is a tensorflow model at this time and we do not need to save as one tensorflow model, but we need to save it as one tensorflow model before we can do some operation
    # # or we just save the bigdl model from the loaded model/
    # # cause this model is keras model, but whether or not we need to create one tensorflow model? yes, we need, so that we cna achieve the tensorflow inference
    # # print("-----------got the model---------")
    # #########################
    # # from bigdl.nn.layer import *
    # # from bigdl.optim.optimizer import *
    # # from bigdl.util.common import *
    #
    # # # create a graph model
    # # linear = Linear(10, 2)()
    # # sigmoid = Sigmoid()(linear)
    # # softmax = SoftMax()(sigmoid)
    # # model = Model([linear], [softmax])
    #
    # # save it to Tensorflow model file
    # ### remember to transfor to nchw from nhwc
    # model.save_tensorflow([("input", [3, 224, 224])], "/tmp/tensorflow_from_bigdl_inception-v1_imagenet_0.4.0_model.pb", data_format="nchw")
    # print("-----------save the model as one tensorflow model type---------")
    #
    # #########################

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
            warmupDelta = (maxlr - options.learningRate) / warmup_iteration
        polyIteration = maxIteration - warmup_iteration
        lrSchedule = SequentialSchedule(iterationPerEpoch)
        lrSchedule.add(Warmup(warmupDelta), warmup_iteration)
        lrSchedule.add(Poly(0.5, polyIteration), polyIteration)
        ## can we use SGD in this way?
        optim = SGD(learningrate=options.learningRate, learningrate_decay=0.0, weightdecay=options.weightDecay,
                    momentum=0.9, dampening=0.0, nesterov=False,
                    leaningrate_schedule=lrSchedule)

    ## where to find the same function
    # create triggers
    if options.maxEpoch:
        checkpoint_trigger = EveryEpoch()
        test_trigger = EveryEpoch()
        end_trigger = MaxEpoch(options.maxEpoch)
    else:
        checkpoint_trigger = SeveralIteration(options.checkpointIteration)
        test_trigger = SeveralIteration(options.checkpointIteration)
        end_trigger = MaxIteration(options.maxIteration)

    ### set classifier
    ### not sure if this can use
    # classifier = NNClassifier(model, CrossEntropyCriterion()) \
    #    .setBatchSize(options.batchSize).setMaxEpoch(70).setFeaturesCol("image") \
    #    .setCachingSample(False).setOptimMethod(optim)
    # pipeline = Pipeline(stages=[classifier])

    # # Optimizer
    # model.compile(optimizer=optim,
    #                 loss='categorical_crossentropy',
    #                 metrics=['accuracy', 'top5acc'])
    # model.fit(x = train_data, batch_size=options.batchSize, nb_epoch=70, validation_data=val_data)

    # trainedModel = pipeline.fit(train_data)
    # predictionDF = trainedModel.transform(val_data).cache()
    # predictionDF.select("name", "label", "prediction").sort("label", ascending=False).show(10)
    # predictionDF.select("name", "label", "prediction").show(10)
    # correct = predictionDF.filter("label=prediction").count()
    # overall = predictionDF.count()
    # accuracy = correct * 1.0 / overall
    # print("Test Error = %g " % (1.0 - accuracy))

    optimizer = TFOptimizer(loss, optim, val_outputs=[logits], val_labels=[labels], val_method=Top1Accuracy())
    optimizer.set_train_summary(TrainSummary("/tmp/logs/inceptionV1", "inceptionV1"))
    optimizer.set_val_summary(ValidationSummary("/tmp/logs/inceptionV1", "inceptionV1"))
    optimizer.optimize(end_trigger=end_trigger)

    # #################################
    # # construct the model from TFDataset
    # images, labels = dataset.tensors
    #
    # with slim.arg_scope(lenet.lenet_arg_scope()):
    #     logits, end_points = lenet.lenet(images, num_classes=10, is_training=True)
    #
    # loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
    #
    # # create a optimizer
    # optimizer = TFOptimizer(loss, Adam(1e-3),
    #                         val_outputs=[logits],
    #                         val_labels=[labels],
    #                         val_method=Top1Accuracy())
    # optimizer.set_train_summary(TrainSummary("/tmp/az_lenet", "lenet"))
    # optimizer.set_val_summary(ValidationSummary("/tmp/az_lenet", "lenet"))
    # # kick off training
    # optimizer.optimize(end_trigger=MaxEpoch(max_epoch))

    saver = tf.train.Saver()
    saver.save(optimizer.sess, "/root/manfei/model")
    #################################

    # optimizer = Optimizer.create(
    #     model=inception_model,
    #     training_set=train_data,
    #     optim_method=optim,
    #     criterion=ClassNLLCriterion(),
    #     end_trigger=end_trigger,
    #     batch_size=options.batchSize
    # )

    # if options.checkpoint:
    #     optimizer.set_checkpoint(checkpoint_trigger, options.checkpoint, options.overwrite)
    #
    # if options.gradientMin and options.gradientMax:
    #     optimizer.set_gradclip_const(options.gradientMin, options.gradientMax)
    #
    # if options.gradientL2NormThreshold:
    #     optimizer.set_gradclip_l2norm(options.gradientL2NormThreshold)

    # optimizer.set_validation(trigger=test_trigger,
    #                          val_rdd=val_data,
    #                          batch_size=options.batchSize,
    #                          val_method=[Top1Accuracy(), Top5Accuracy()])

    # trained_model = optimizer.optimize()

    sc.stop()
