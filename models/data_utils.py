# data_utils.py
import pyrealsense2 as rs

def initialize_camera():
    # カメラパイプラインの設定
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 深度とカラーストリームを有効化
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # IMUデータストリームの有効化
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    
    # カメラのパイプラインを開始
    pipeline.start(config)
    return pipeline

def get_depth_data(pipeline):
    # フレームセットの取得
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    
    # 深度データの取得
    depth_image = np.asanyarray(depth_frame.get_data())
    return depth_image

def get_imu_data(pipeline):
    # フレームセットの取得
    frames = pipeline.wait_for_frames()
    accel_frame = frames.first(rs.stream.accel)
    gyro_frame = frames.first(rs.stream.gyro)
    
    if not accel_frame or not gyro_frame:
        return None, None
    
    # 加速度とジャイロデータの取得
    accel = accel_frame.as_motion_frame().get_motion_data()
    gyro = gyro_frame.as_motion_frame().get_motion_data()
    
    return accel, gyro
