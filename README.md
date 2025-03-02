# hik_driver 
> 海康相机采图
使用例子如下，设置保存地址、曝光、增益、是否自动采图

  ` python3.8 kaca.py --save_directory "saved_images" --exposure_time 20000 --gain 18.5 --auto_capture True    `

或者：

  ` python3.8 kaca.py  `

手动拍照时回车保存照片，自动拍照时每5秒拍一张。有自动命名，不必担心图片名字重合

# 环境配置
>opencv
>
>海康驱动
>
>python3.8  或  python3.10（没试过别的）

## 记得设置输出图片格式为rgb8!!!!!
