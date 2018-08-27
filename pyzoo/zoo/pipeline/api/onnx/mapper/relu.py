# See the License for the specific language governing permissions and
# limitations under the License.
#
from zoo.pipeline.api.onnx.mapper.operator_mapper import OperatorMapper
from zoo.pipeline.api.onnx.onnx_helper import OnnxHelper
import zoo.pipeline.api.keras.layers as zlayers

class ReluMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(ReluMapper, self).__init__(node, _params, _all_tensors)

    def create_operator(self):
        assert len(self.inputs) == 1, "Conv accept single input only"
        rank = len(self.inputs[0].get_input_shape())

        if (rank == 2):  # NCHW

            relu = zlayers.Activation("relu")
            return relu

        else:
            raise Exception("not supported.")