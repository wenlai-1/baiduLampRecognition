import cv2
import numpy as np
from cnocr import CnOcr

def detect_colors(image_path):
    # 读取图像
    image = (image_path)
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
    results = []
    if has_yellow:
        results.append("黄灯")
    elif has_red:
        results.append("红灯")
    elif has_green:
        results.append("绿灯")

    if results:
        return f"当前为{', '.join(results)}"
    else:
        return "未检测到"
    
# 读取图像
image = cv2.imread('C:/Users/ROG/Desktop/honglvdeng/3.jpg')  # 替换为你的图像路径
if image is None:
    print("Error: 图像文件未找到")
    exit()

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
    exit()

# 找到最大轮廓（即最大的黑色区域）
largest_contour = max(contours, key=cv2.contourArea)

# 检查最大轮廓的面积是否大于10000
#print(f"最大轮廓的面积是: {cv2.contourArea(largest_contour)}")
if cv2.contourArea(largest_contour) <= 10000:
    #print("未检测到红绿灯")
    exit()

# 创建一个空白图像来绘制轮廓
mask = np.zeros_like(lower_half)

# 绘制最大轮廓
cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)

# 应用掩码来显示只有最大黑色区域的图像
masked_image = cv2.bitwise_and(lower_half, mask)
x, y, w, h = cv2.boundingRect(largest_contour)
black_region = masked_image[y:y+h, x:x+w]

# 去掉black_region的左边125的部分
left_x = 125
cropped_black_region = black_region[:, left_x:]

# 显示结果
# cv2.imshow("Masked Image", masked_image)
image_path = masked_image
print(detect_colors(image_path))

ocr = CnOcr(det_model_name='naive_det') 
out = ocr.ocr(cropped_black_region)

# 假设 out 是一个列表，且每个子元素包含一个字符串
try:
    #假设 out 的结构是 [[("文字", 置信度, 其他)], ...]
    out_str = ''.join([item['text'] if isinstance(item, dict) else item[0] for item in out])
except (KeyError, IndexError, TypeError):
    print("OCR 结果的结构与预期不符")
    out_str = ""

# 替换 OCR 结果中的 "日" 为 "8"
out_str = out_str.replace("日", "8")

# 打印替换后的字符串
print(out_str)

cv2.imshow("Black Region", black_region)
cv2.waitKey(0)
cv2.destroyAllWindows()