#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Operator attributes conversion"""
from .ops_converter import conv, batch_norm, relu, max_pool, _sum, average_pool
from .ops_converter import reshape, gemm, softmax, constant, shape, gather
from .ops_converter import unsqueeze, concat


# convert_map defines maps of operator names to converter functor(callable)
# defined in the op_translations module.
_convert_map = {
    # Generator Functions
    'Constant'          : constant,
    # 'RandomUniform'     : random_uniform,
    # 'RandomNormal'      : random_normal,
    # 'RandomUniformLike' : random_uniform,
    # 'RandomNormalLike'  : random_normal,
    # 'Multinomial'       : sample_multinomial,
    # # Arithmetic Operators
    # 'Add'               : add,
    # 'Sub'               : subtract,
    # 'Mul'               : multiply,
    # 'Div'               : divide,
    # 'Abs'               : absolute,
    # 'Neg'               : negative,
    'Sum'               : _sum, #elemwise sum
    # #Hyperbolic functions
    # 'Tanh'              : tanh,
    # # Rounding
    # 'Ceil'              : ceil,
    # 'Floor'             : floor,
    # # Joining and spliting
    'Concat'            : concat,
    # # Basic neural network functions
    # 'Sigmoid'           : sigmoid,
    'Relu'              : relu,
    # 'Pad'               : pad,
    # 'MatMul'            : matrix_multiplication, #linalg_gemm2
    'Conv'              : conv,
    # 'ConvTranspose'     : deconv,
    'BatchNormalization': batch_norm,
    # 'SpatialBN'         : batch_norm,
    # 'LeakyRelu'         : leaky_relu,
    # 'Elu'               : _elu,
    # 'PRelu'             : _prelu,
    # 'Selu'              : _selu,
    'Softmax'           : softmax,
    # 'FC'                : fully_connected,
    # 'GlobalAveragePool' : global_avgpooling,
    # 'GlobalMaxPool'     : global_maxpooling,
    # 'GlobalLpPool'      : global_lppooling,
    'Gemm'              : gemm,
    # 'LRN'               : local_response_norm,
    # 'Dropout'           : dropout,
    # # Changing shape and type.
    'Reshape'           : reshape,
    # 'Cast'              : cast,
    # 'Split'             : split,
    # 'Slice'             : _slice,
    # 'Transpose'         : transpose,
    # 'Squeeze'           : squeeze,
    'Unsqueeze'         : unsqueeze,
    # 'Flatten'           : flatten,
    # 'Identity'          : identity,
    # #Powers
    # 'Reciprocal'        : reciprocal,
    # 'Sqrt'              : squareroot,
    # 'Pow'               : power,
    # 'Exp'               : exponent,
    # 'Log'               : _log,
    # # Reduce Functions
    # 'ReduceMax'         : reduce_max,
    # 'ReduceMean'        : reduce_mean,
    # 'ReduceMin'         : reduce_min,
    # 'ReduceSum'         : reduce_sum,
    # 'ReduceProd'        : reduce_prod,
    'AveragePool'       : average_pool,
    'MaxPool'           : max_pool,
    # # Sorting and Searching
    # 'ArgMax'            : argmax,
    # 'ArgMin'            : argmin,
    # 'Max'               : maximum,
    # 'Min'               : minimum,
    # 'Clip'              : clip,
    # 'ReduceLogSum'      : reduce_log_sum,
    # 'ReduceLogSumExp'   : reduce_log_sum_exp,
    # 'ReduceSumSquare'   : reduce_sum_square,
    # 'ReduceL1'          : reduce_l1,
    # 'ReduceL2'          : reduce_l2,
    # 'MaxRoiPool'        : max_roi_pooling,
    # 'InstanceNormalization' : instance_norm,
    # 'LogSoftmax'        : log_softmax,
    # 'Softsign'          : softsign,
    # 'Less'              : lesser,
    # 'Greater'           : greater,
    # 'Equal'             : equal,
    # 'And'               : logical_and,
    # 'Xor'               : logical_xor,
    # 'Not'               : logical_not,
    # 'Or'                : logical_or,
    # 'Mean'              : mean,
    # 'Acos'              : arccos,
    # 'Asin'              : arcsin,
    # 'Atan'              : arctan,
    # 'Cos'               : _cos,
    # 'Sin'               : _sin,
    # 'Softplus'          : softplus,
    # 'Tan'               : _tan,
    'Shape'             : shape,
    # 'Size'              : size,
    'Gather'            : gather,
    # 'HardSigmoid'       : hardsigmoid,
    # 'LpPool'            : lp_pooling,
    # 'DepthToSpace'      : depthtospace,
    # 'SpaceToDepth'      : spacetodepth,
    # 'Hardmax'           : hardmax,
    # 'LpNormalization'   : lpnormalization
}
