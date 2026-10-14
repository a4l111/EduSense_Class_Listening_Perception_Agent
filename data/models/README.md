# 服务器侧 ONNX 模型目录

真实 gRPC 推理服务运行前，请放入：

- `yolov8n.onnx`
- `movenet_lightning.onnx`

Optional for person re-identification only:

- `osnet_x0_25_slim.onnx`

Education-scene pipeline optional models:

- `yolov8n-face.onnx` or another YOLO-like face detector
- `expression.onnx` with labels: focused, confused, tired, distracted, neutral
- `head_pose.onnx` with yaw/pitch/roll output
- `gaze.onnx` with either screen/book/phone/away logits or yaw/pitch output

The server can start without these optional models. In that case
`--enable-education` uses geometry-based fallback estimates so the end-to-end
pipeline remains runnable during integration.

源码包不包含这些模型文件。
