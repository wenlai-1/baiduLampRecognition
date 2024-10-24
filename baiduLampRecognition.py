import os
import subprocess

from cnocr import CnOcr

from time import sleep

import numpy as np
#from appium import webdriver
#from appium.options.android import UiAutomator2Options
#from appium.webdriver.common.appiumby import AppiumBy
import cv2

# from baiduditu import image


def adb_tap(x, y):
    '''
    adb模拟点击
    :param x: 横坐标
    :param y: 纵坐标
    :return:
    '''
    cmd = f'adb shell input tap {x} {y}'
    os.system(cmd)

# 下面四个函数是我基于初版修改的在这个基础上修改

def crop_lamp_area(image):

    '''
    检测红绿灯区域，并抠图
    :param image: 输入图片对象
    :return: 返回的图像
    '''

    # 获取图像尺寸
    height, width = image.shape[:2]

    # 只处理图像的下四分之三部分
    lower_half = image[height//4*1:, :]

    # 转换为灰度图
    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)

    # 进行二值化处理，假设黑色为低于100的像素
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到轮廓，输出提示
    if len(contours) == 0:
        print("没有找到黑色区域")
        return None

    # 找到最大轮廓（即最大的黑色区域）
    largest_contour = max(contours, key=cv2.contourArea)

    # 检查最大轮廓的面积是否大于5000
    #print(f"最大轮廓的面积是: {cv2.contourArea(largest_contour)}")
    if cv2.contourArea(largest_contour) <= 5000:
        print("未检测到红绿灯")
        return None

    # 创建一个空白图像来绘制轮廓
    mask = np.zeros_like(lower_half)

    # 绘制最大轮廓
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)

    # 应用掩码来显示只有最大黑色区域的图像
    masked_image = cv2.bitwise_and(lower_half, mask)
    x, y, w, h = cv2.boundingRect(largest_contour)
    black_region = masked_image[y:y+h, x:x+w]


    # 显示结果
    cv2.imshow("Black Region", black_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return black_region

def detect_lamp_colors(image):

    # 转换到HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义颜色范围
    green_lower = np.array([35, 100, 50])
    green_upper = np.array([85, 255, 255])
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([20, 100, 50])
    yellow_upper = np.array([30, 255, 255])

    # 创建颜色掩码
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
    red_mask = cv2.add(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    # 判断是否检测到颜色
    has_green = np.any(green_mask)
    has_red = np.any(red_mask)
    has_yellow = np.any(yellow_mask)

    # 输出结果

    if has_yellow:
        print("黄灯")
        return 0
    elif has_red:
        print("红灯")
        return -1
    elif has_green:
        print("绿灯")
        return 1
    else:
        print("未检测到")
        return None

def detect_lamp_countdown(black_region):

    ##图像预处理
    # 去掉black_region的左边57的部分和右边15的部分
    left_x = 57
    right_x = 15
    cropped_black_region = black_region[:, left_x:-right_x]
    # 将裁剪后的图像转换为灰度图像
    gray_image = cv2.cvtColor(cropped_black_region, cv2.COLOR_BGR2GRAY)
    # 应用高斯模糊(目前最优的参数)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    # 创建一个形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 应用膨胀
    dilated_image = cv2.dilate(blurred_image, kernel, iterations=1)

    ocr = CnOcr(det_model_name='naive_det') 
    out = ocr.ocr(dilated_image)

    # 假设 out 是一个列表，且每个子元素包含一个字符串
    try:
        #假设 out 的结构是 [[("文字", 置信度, 其他)], ...]
        out_str = ''.join([item['text'] if isinstance(item, dict) else item[0] for item in out])
    except (KeyError, IndexError, TypeError):
        print("OCR 结果的结构与预期不符")
        out_str = ""

    # 替换 OCR 结果：
    out_str = out_str.replace("日", "8").replace("（门（（", "10").replace("马", "9").replace("{", "1").replace("B", "8").replace("|", "1").replace("S", "5").replace("I", "1").replace("D", "0")
    # 打印替换后的字符串
    print(out_str)
    return out_str

def baiduLampRecognition(image):

    '''
    喂入一张截图，输出红绿灯倒计时
    :param image:
    :return:lampColor:信号灯颜色
            lampCountdown:信号灯倒计时数字
    '''

    # 抠小图
    lampArea = crop_lamp_area(image)

    # 识别颜色
    lampColor = detect_lamp_colors(lampArea)

    #
    if lampColor == None:
        # 灯的颜色都没检测到，也不用检测数字了直接返回None
        return None, None
    else:
        # 识别灯的倒计时数字
        lampCountdown = detect_lamp_countdown(lampArea)

        return lampColor, lampCountdown

# 上面四个函数是我基于初版修改的在这个基础上修改

def lampImageAcquire():

    '''
    设定某路口坐标 并截取该位置图片
    :return:
    '''

    # 测试配置
    capabilities = {
        'platformName': 'Android',
        'automationName': 'uiautomator2',
        'deviceName': 'Android',
        'appPackage': 'com.baidu.BaiduMap',
        'appActivity': 'com.baidu.baidumaps.MapsActivity',
        'noReset': 'true',
        'unicodeKeyboard': 'true',
        # 'geoLocation': {'latitude': 29.557259, 'longitude': 106.577052}  # 设置模拟位置
    }



    appium_server_url = 'http://localhost:4723'

    # 加载要测试的app
    driver = webdriver.Remote(appium_server_url, options=UiAutomator2Options().load_capabilities(capabilities))

    # print("ok")
    # 模拟位置
    driver.set_location(29.46295436497, 106.52360420624, 100)
    # driver.set_location(29.565621, 106.47669, 100)
    # 点击百度地图输入框

    search_box = driver.find_element(AppiumBy.ID, 'com.baidu.BaiduMap:id/common_search_box_home')
    search_box.click()  # 如果需要先点击打开输入框

    # 输入目标位置
    input_box = driver.find_element(AppiumBy.ID, 'com.baidu.BaiduMap:id/tvSearchBoxInput')
    input_box.send_keys('重庆邮电大学')

    sleep(1)

    # 点击搜索按钮
    driver.find_element(AppiumBy.ID, 'com.baidu.BaiduMap:id/tvSearchButton').click()

    # 单击到这去

    # class_text='className("android.view.ViewGroup").text("重庆邮电大学")'
    # driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR, class_text).click()
    # driver.find_element(AppiumBy.XPATH, '//*[@text="重庆邮电大学"]').click()

    sleep(3)
    adb_tap(376, 543)

    sleep(3)

    # 开始导航
    driver.find_element(AppiumBy.ID, 'com.baidu.BaiduMap:id/to_pro_nav').click()

    sleep(5)

    image = driver.get_screenshot_as_png()

    driver.quit()

    image_array = np.frombuffer(image, dtype=np.uint8)

    # 使用 OpenCV 解码图像
    image_cv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # 显示图像
    # cv2.imshow('Screenshot', image_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image_cv

def check_adb_devices():
    '''
    检查adb 设备，并返回设备sn list

    :return: 设备sn list
    '''
    adb_list = []
    ret = os.popen('adb devices').readlines()
    # print('ret={}'.format(ret))
    if len(ret) == 0:
        print('未读取到信息')
        # return adb_list
        return False
    else:
        for n in ret:
            if '\tdevice\n' in n:
                adb = str(n).strip().split('\tdevice')[0].strip()
                adb_list.append(str(adb))

        if len(adb_list) == 1:

            # print('adb设备数量={}，adb_list={}'.format(len(adb_list), adb_list))
            # return adb_list
            print('已正确识别到adb 设备...')
            return True

        print('未识别到恰当数量的adb 设备...')
        return False

def initEnvironment():
    '''
    初始化环境
    包括启动安卓模拟器、启动appium服务、确保ADB通畅、与CPP程序管道通信通畅。保证后续操作顺滑
    :return:
    '''

    isAndroidOnline = False

    while isAndroidOnline == False:

        #死循环轮询设备
        isAndroidOnline = check_adb_devices()
        sleep(1)

    if subprocess.Popen('cmd /K appium', creationflags=subprocess.CREATE_NEW_CONSOLE):
        print("已启动appium服务")
        print("安卓测试平台初始化完成")
        return True
    else:
        print("安卓测试平台初始化失败！")
        return False






if __name__ == '__main__':


    image1 = cv2.imread("C:/Users/ROG/Desktop/baiduditu/45.png")
    lampColor, lampCountdown = baiduLampRecognition(image1)
