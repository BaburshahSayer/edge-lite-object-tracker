import onnxruntime as ort

model_path = "/Users/babursayer/Desktop/0101/edge-lite-object-tracker/models/yolov5s.onnx"
session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

print("Input name:", input_name)
print("Input shape:", input_shape)
print("Input type:", input_type)

output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape
output_type = session.get_outputs()[0].type

print("Output name:", output_name)
print("Output shape:", output_shape)
print("Output type:", output_type)
