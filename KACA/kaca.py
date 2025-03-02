import time
import argparse
from hik_camera import call_back_get_image, start_grab_and_get_data_size, close_and_destroy_device, set_Value, \
    get_Value, image_control
from MvImport.MvCameraControl_class import *
import cv2
import numpy as np
import os
import re

# 从命令行参数读取配置
def parse_args():
    parser = argparse.ArgumentParser(description="Hikvision Camera Capture")

    # 添加命令行参数
    #eg: python3.8 kaca.py --save_directory "saved_images" --exposure_time 20000 --gain 18.5 --auto_capture True
    parser.add_argument('--save_directory', type=str, default='saved_images', help='保存图像的目录')
    parser.add_argument('--exposure_time', type=float, default=16000, help='曝光时间')
    parser.add_argument('--gain', type=float, default=17.9, help='增益')
    parser.add_argument('--auto_capture', type=bool, default=False, help='是否自动拍照模式（默认：否）')

    return parser.parse_args()

# 加载命令行参数
args = parse_args()

# 初始化保存路径和文件编号
save_directory = args.save_directory  # 从命令行参数获取保存目录
if not os.path.exists(save_directory):
    os.makedirs(save_directory)  # 如果目录不存在，创建目录

# 扫描目录并初始化最大编号
def get_max_image_counter():
    max_counter = 0
    pattern = re.compile(r"rasgo-6(\d{4})\.jpg")  # 匹配文件名中的数字部分

    # 扫描目录中的所有文件并找到最大的编号
    for filename in os.listdir(save_directory):
        match = pattern.match(filename)
        if match:
            # 提取文件名中的数字部分并更新最大编号
            counter = int(match.group(1))
            if counter > max_counter:
                max_counter = counter

    return max_counter

# 初始化递增编号
image_counter = get_max_image_counter() + 1  # 从最大的编号的下一个开始

def save_image(image):
    global image_counter
    # 递增编号并保存图片
    image_filename = os.path.join(save_directory, f"rasgo-6{image_counter:04d}.jpg")
    cv2.imwrite(image_filename, image)
    print(f"Original image saved as '{image_filename}'.")
    image_counter += 1  # 递增编号

def hik_camera_get(auto_capture=False):
    global camera_image, original_image, image_counter
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # 枚举设备
    while True:
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print(f"Enum devices fail! ret[0x{ret:x}]")
            time.sleep(1)
            continue

        if deviceList.nDeviceNum == 0:
            print("No devices found!")
            time.sleep(1)
            continue
        else:
            print(f"Found {deviceList.nDeviceNum} devices!")
            break

    for i in range(deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print(f"Found GigE device: [{i}]")
            # 输出设备信息...
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print(f"Found USB device: [{i}]")
            # 输出设备信息...

    # 选择设备
    nConnectionNum = 0
    if nConnectionNum >= deviceList.nDeviceNum:
        print("Input error!")
        return

    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[nConnectionNum], POINTER(MV_CC_DEVICE_INFO)).contents

    # 创建相机句柄
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print(f"Create handle fail! ret[0x{ret:x}]")
        return

    # 打开设备
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"Open device fail! ret[0x{ret:x}]")
        return

    print("Device opened successfully.")

    # 获取相机的参数（例如曝光时间、增益等）
    exposure_time = args.exposure_time  # 从命令行参数获取曝光时间
    gain = args.gain  # 从命令行参数获取增益

    set_Value(cam, param_type="float_value", node_name="ExposureTime", node_value=exposure_time)
    set_Value(cam, param_type="float_value", node_name="Gain", node_value=gain)

    # 开始抓取图像
    start_grab_and_get_data_size(cam)
    stParam = MVCC_INTVALUE_EX()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
    ret = cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
    if ret != 0:
        print(f"Get payload size fail! ret[0x{ret:x}]")
        return

    nDataSize = stParam.nCurValue
    pData = (c_ubyte * nDataSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

    # 主循环，抓取并显示图像
    last_capture_time = time.time()

    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            print(f"Frame received, size: {len(pData)}")
            image = np.frombuffer(pData, dtype=np.uint8)
            camera_image = image.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, 3)  # 根据相机格式调整

            # 保存原始图像
            original_image = camera_image.copy()

            # 缩小图像（1/2）
            small_image = cv2.resize(camera_image, (stFrameInfo.nWidth // 2, stFrameInfo.nHeight // 2))

            # 显示图像
            print(f"Image shape: {small_image.shape if small_image is not None else 'None'}")
            cv2.imshow("Hikvision Camera Image (Small)", small_image)

            # 如果是自动拍照模式，检查每5秒拍照一次
            if auto_capture and (time.time() - last_capture_time >= 5):
                save_image(original_image)  # 保存原始图像
                last_capture_time = time.time()  # 更新上次拍照时间

            # 检测按键，按回车键保存图像
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按 'q' 退出
                break
            elif key == 13:  # 按回车键保存原图
                save_image(original_image)  # 保存原始图像
                last_capture_time = time.time()  # 更新上次拍照时间

        else:
            print(f"No data[0x{ret:x}]")

    # 关闭设备
    close_and_destroy_device(cam)

# 主程序入口
if __name__ == "__main__":
    camera_mode = 'hik'  # 'test':图片测试, 'video':视频测试, 'hik':海康相机, 'usb':USB相机
    camera_image = None
    original_image = None  # 存储原始图像

    if camera_mode == 'test':
        camera_image = cv2.imread('models/test.jpg')
    elif camera_mode == 'hik':
        # 启动相机并传入参数
        hik_camera_get(auto_capture=args.auto_capture)  # 传递自动拍照模式的参数

    if camera_image is None:
        print("Waiting for image...")
