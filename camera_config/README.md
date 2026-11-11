# 相机参数配置

在 parameters.json 中填写参数，test_main 每次打开相机后、开始采集前自动加载。
null 表示不修改该项，继续使用相机当前值。

- ExposureTime：曝光时间，单位微秒。例如 10000 表示 10 ms；填写后自动关闭自动曝光。
- Gain：增益，单位和范围由相机型号决定。旧型号可能使用 GainRaw 原始单位；填写后自动关闭自动增益。
- AcquisitionFrameRate：相机采集帧率，单位帧/秒；填写后启用帧率控制。实际帧率也受曝光时间和传输速度限制。
- Width、Height：采集宽度和高度，单位像素，需符合相机的范围及步长，并受当前 ROI 偏移限制。

例如（仅示例，需要根据实际相机填写）：

```json
{
  "ExposureTime": 10000.0,
  "Gain": 0,
  "AcquisitionFrameRate": 10.0,
  "Width": null,
  "Height": null
}
```

启动时会打印已设置的参数及采集分辨率。参数不受支持、不可写或超出范围时会报错并关闭相机，避免在错误设置下继续采集。硬件写入不具备事务性：报错前已经成功写入的参数可能保留在相机中。

修改后重启程序生效。采集帧率与 CSV 数据汇总间隔是分别设置的。

其他程序可在 camera.Open() 后调用：

```python
from camera_config import apply_camera_config
apply_camera_config(camera)
```
