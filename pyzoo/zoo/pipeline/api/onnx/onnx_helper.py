#
# Copyright 2018 Analytics Zoo Authors.
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
import warnings


class OnnxHelper:
    """
    .. note:: `zoo.pipeline.api.onnx` is deprecated in 0.10.0
    This will be removed in future releases.
    """
    @staticmethod
    def parse_attr(attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        warnings.warn("deprecated in 0.10.0, and will be removed in future release")
        attrs = {}
        for a in attr_proto:
            for f in ['f', 'i', 's']:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ['floats', 'ints', 'strings']:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ['t', 'g']:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ['tensors', 'graphs']:
                if list(getattr(a, f)):
                    raise NotImplementedError("Filed {} is not supported in mxnet.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs

    @staticmethod
    def to_numpy(tensor_proto):
        """Grab data in TensorProto and to_tensor to numpy array."""
        warnings.warn("deprecated in 0.10.0, and will be removed in future release")
        try:
            from onnx.numpy_helper import to_array
        except ImportError as e:
            raise ImportError("Unable to import onnx which is required {}".format(e))
        np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
        return np_array

    @staticmethod
    def get_shape_from_node(valueInfoProto):
        warnings.warn("deprecated in 0.10.0, and will be removed in future release")
        return [int(dim.dim_value) for dim in valueInfoProto.type.tensor_type.shape.dim]

    @staticmethod
    def get_padds(onnx_attr):
        warnings.warn("deprecated in 0.10.0, and will be removed in future release")
        border_mode = None
        pads = None

        if "auto_pad" in onnx_attr.keys():
            if onnx_attr['auto_pad'].decode() == 'SAME_UPPER':
                border_mode = 'same'
            elif onnx_attr['auto_pad'].decode() == 'VALID':
                border_mode = 'valid'
            elif onnx_attr['auto_pad'].decode() == 'NOTSET':
                assert "pads" in onnx_attr.keys(), "you should specify pads explicitly"
            else:
                raise NotImplementedError('Unknown auto_pad mode "%s", '
                                          'only SAME_UPPER and VALID are supported.'
                                          % onnx_attr['auto_pad'])

        # In ONNX, pads format is [x1_begin, x2_begin...x1_end, x2_end,...].
        # While pads format we supported should be [x1_begin, x1_end, x2_begin, x2_end...]
        if "pads" in onnx_attr.keys():
            pads = [int(i) for i in onnx_attr["pads"]]
            if len(pads) == 4:
                assert pads[0] == pads[2]
                assert pads[1] == pads[3]
                pads = [pads[0], pads[1]]
            elif len(pads) == 2:
                assert pads[0] == pads[1]
                pads = pads[0]

        return border_mode, pads
