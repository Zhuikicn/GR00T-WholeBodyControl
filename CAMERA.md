### 启动pyrealsense 指南
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera realsense \
    --ego-view-device-id 406122070774 \
    --port 5555
### 启动zed相机
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera zed \
    --port 5555