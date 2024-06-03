from PIL import Image, ImageDraw
import os

# 打开图像
category = 'backpack'
angle = '270.0'


method_1 = 'tensorf'
folder_1 = ''

method_3 = 'gt'
folder_3 = ''

method_4 = 'intratomo'
folder_4 = ''

method_5 = 'Lineformer'
folder_5 = ''

method_6 = 'naf'
folder_6 = ''

method_7 = 'neat'
folder_7 = ''

method_2 = 'nerf'
folder_2 = ''




image_1 = Image.open(f"all_results_proj/{category}/{method_1}/{folder_1}/angle_{angle}.png")  
image_2 = Image.open(f"all_results_proj/{category}/{method_2}/{folder_2}/angle_{angle}.png")  
image_3 = Image.open(f"all_results_proj/{category}/{method_3}/{folder_3}/angle_{angle}.png")
image_4 = Image.open(f"all_results_proj/{category}/{method_4}/{folder_4}/angle_{angle}.png")  
image_5 = Image.open(f"all_results_proj/{category}/{method_5}/{folder_5}/angle_{angle}.png")  
image_6 = Image.open(f"all_results_proj/{category}/{method_6}/{folder_6}/angle_{angle}.png") 
image_7 = Image.open(f"all_results_proj/{category}/{method_7}/{folder_7}/angle_{angle}.png")  

# 切掉多余的白边
# 长: 830, 高: 630
box = (20, 150, 850, 780)

image_1_crop = image_1.crop(box)
image_2_crop = image_2.crop(box)
image_3_crop = image_3.crop(box)
image_4_crop = image_4.crop(box)
image_5_crop = image_5.crop(box)
image_6_crop = image_6.crop(box)
image_7_crop = image_7.crop(box)

# image_1_crop = image_1
# image_2_crop = image_2
# image_3_crop = image_3
# image_4_crop = image_4
# image_5_crop = image_5
# image_6_crop = image_6
# image_7_crop = image_7


# 保存没有bbox的图像
image_3_crop.save(f"draw_bbox/{category}/{method_3}_no_bbox.png")

# 创建一个可以在图像上绘图的对象
draw_1 = ImageDraw.Draw(image_1_crop)
draw_2 = ImageDraw.Draw(image_2_crop)
draw_3 = ImageDraw.Draw(image_3_crop)
draw_4 = ImageDraw.Draw(image_4_crop)
draw_5 = ImageDraw.Draw(image_5_crop)
draw_6 = ImageDraw.Draw(image_6_crop)
draw_7 = ImageDraw.Draw(image_7_crop)

w = 124
h = 90

# 定义矩形框的位置和颜色
left = 300 - 22  # 左上角横坐标
top = 176 - 150  # 左上角纵坐标
right = left + w  # 右下角横坐标
bottom = top + h  # 右下角纵坐标
rectangle_color = (0, 0, 255)  # 蓝色 (R, G, B)
rectangle_color = 'blue'

## red bbox
left_red = 720 - 22  # 左上角横坐标
top_red = 505 - 150 # 左上角纵坐标
right_red = left_red + w  # 右下角横坐标
bottom_red = top_red + h  # 右下角纵坐标
rectangle_color_red = 'red'


# 剪切矩形框中的部分
cropped_image_1 = image_1_crop.crop((left, top, right, bottom))
cropped_image_2 = image_2_crop.crop((left, top, right, bottom))
cropped_image_3 = image_3_crop.crop((left, top, right, bottom))
cropped_image_4 = image_4_crop.crop((left, top, right, bottom))
cropped_image_5 = image_5_crop.crop((left, top, right, bottom))
cropped_image_6 = image_6_crop.crop((left, top, right, bottom))
cropped_image_7 = image_7_crop.crop((left, top, right, bottom))

cropped_image_1_red = image_1_crop.crop((left_red, top_red, right_red, bottom_red))
cropped_image_2_red = image_2_crop.crop((left_red, top_red, right_red, bottom_red))
cropped_image_3_red = image_3_crop.crop((left_red, top_red, right_red, bottom_red))
cropped_image_4_red = image_4_crop.crop((left_red, top_red, right_red, bottom_red))
cropped_image_5_red = image_5_crop.crop((left_red, top_red, right_red, bottom_red))
cropped_image_6_red = image_6_crop.crop((left_red, top_red, right_red, bottom_red))
cropped_image_7_red = image_7_crop.crop((left_red, top_red, right_red, bottom_red))






save_dir = f'draw_bbox/{category}/'
os.makedirs(save_dir,exist_ok=True)


# 保存剪切后的图像
cropped_image_1.save(f"draw_bbox/{category}/{method_1}_crop.png")
cropped_image_2.save(f"draw_bbox/{category}/{method_2}_crop.png")
cropped_image_3.save(f"draw_bbox/{category}/{method_3}_crop.png")
cropped_image_4.save(f"draw_bbox/{category}/{method_4}_crop.png")
cropped_image_5.save(f"draw_bbox/{category}/{method_5}_crop.png")
cropped_image_6.save(f"draw_bbox/{category}/{method_6}_crop.png")
cropped_image_7.save(f"draw_bbox/{category}/{method_7}_crop.png")

cropped_image_1_red.save(f"draw_bbox/{category}/{method_1}_crop_red.png")
cropped_image_2_red.save(f"draw_bbox/{category}/{method_2}_crop_red.png")
cropped_image_3_red.save(f"draw_bbox/{category}/{method_3}_crop_red.png")
cropped_image_4_red.save(f"draw_bbox/{category}/{method_4}_crop_red.png")
cropped_image_5_red.save(f"draw_bbox/{category}/{method_5}_crop_red.png")
cropped_image_6_red.save(f"draw_bbox/{category}/{method_6}_crop_red.png")
cropped_image_7_red.save(f"draw_bbox/{category}/{method_7}_crop_red.png")



width = 5
# 绘制矩形框
draw_1.rectangle([left, top, right, bottom], outline=rectangle_color, width=width)
draw_2.rectangle([left, top, right, bottom], outline=rectangle_color, width=width)
draw_3.rectangle([left, top, right, bottom], outline=rectangle_color, width=width)
draw_4.rectangle([left, top, right, bottom], outline=rectangle_color, width=width)
draw_5.rectangle([left, top, right, bottom], outline=rectangle_color, width=width)
draw_6.rectangle([left, top, right, bottom], outline=rectangle_color, width=width)
draw_7.rectangle([left, top, right, bottom], outline=rectangle_color, width=width)

draw_1.rectangle([left_red, top_red, right_red, bottom_red], outline=rectangle_color_red, width=width)
draw_2.rectangle([left_red, top_red, right_red, bottom_red], outline=rectangle_color_red, width=width)
draw_3.rectangle([left_red, top_red, right_red, bottom_red], outline=rectangle_color_red, width=width)
draw_4.rectangle([left_red, top_red, right_red, bottom_red], outline=rectangle_color_red, width=width)
draw_5.rectangle([left_red, top_red, right_red, bottom_red], outline=rectangle_color_red, width=width)
draw_6.rectangle([left_red, top_red, right_red, bottom_red], outline=rectangle_color_red, width=width)
draw_7.rectangle([left_red, top_red, right_red, bottom_red], outline=rectangle_color_red, width=width)

# 剪切矩形框中的部分
# 保存带有绘制的矩形框的原图像


image_with_rectangle_1 = image_1_crop.copy()
image_with_rectangle_1.save(f"draw_bbox/{category}/{method_1}.png")

image_with_rectangle_2 = image_2_crop.copy()
image_with_rectangle_2.save(f"draw_bbox/{category}/{method_2}.png")

image_with_rectangle_3 = image_3_crop.copy()
image_with_rectangle_3.save(f"draw_bbox/{category}/{method_3}.png")

image_with_rectangle_4 = image_4_crop.copy()
image_with_rectangle_4.save(f"draw_bbox/{category}/{method_4}.png")

image_with_rectangle_5 = image_5_crop.copy()
image_with_rectangle_5.save(f"draw_bbox/{category}/{method_5}.png")

image_with_rectangle_6 = image_6_crop.copy()
image_with_rectangle_6.save(f"draw_bbox/{category}/{method_6}.png")

image_with_rectangle_7 = image_7_crop.copy()
image_with_rectangle_7.save(f"draw_bbox/{category}/{method_7}.png")
