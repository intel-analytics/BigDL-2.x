from zoo.serving.client import InputQueue, OutputQueue
import numpy as np
# input tensor
x = np.array([2, 3], dtype=np.float32)

input_api = InputQueue()
input_api.enqueue_tensor('my_input', x)

output_api = OutputQueue()
img1_result = output_api.query('my_input')
print("Result is :" + str(img1_result))
