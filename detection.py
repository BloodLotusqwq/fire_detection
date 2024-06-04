import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 设置图片的尺寸和批量大小
img_width, img_height = 150, 150
batch_size = 32


def read_image(image_path):
    """读取图像并转换为RGB格式"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def convert_to_hsv(image):
    """将图像转换为HSV颜色空间"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv_image

def apply_color_threshold(hsv_image):
    """应用颜色阈值分割火焰区域"""
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    return mask

def morphological_operations(mask):
    """形态学操作：膨胀和腐蚀"""
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded

def connected_components_analysis(mask, image):
    """连通组件分析并绘制火焰区域"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):  # 0 是背景，不绘制
        x, y, w, h, area = stats[i]
        if area > 100:  # 过滤掉小区域
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

def display_images(images, titles):
    """显示图像列表"""
    plt.figure(figsize=(20, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.title(title)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.axis('off')
    plt.show()

def fire_detection_pipeline(image_path):
    """火焰检测流水线，显示每一步结果"""
    original_image = read_image(image_path)
    hsv_image = convert_to_hsv(original_image)
    mask = apply_color_threshold(hsv_image)
    morph_image = morphological_operations(mask)
    result_image = connected_components_analysis(morph_image, original_image.copy())

    # 显示每一步结果
    images = [original_image, hsv_image, mask, morph_image, result_image]
    titles = ['Original Image', 'HSV Image', 'Color Threshold', 'Morphological Operations', 'Fire Detection']
    display_images(images, titles)

# 测试函数
fire_detection_pipeline(r'F:\pythonproject\fire_detection1\Fire-Detection-Image-Dataset\Images\Fire_Images\800px-Fires_cross_a_hill_in_SoCal_October_2007.jpg')

def create_image_generators(data_path, img_width, img_height, batch_size):
    """创建训练和验证图像生成器"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 自动分割 20% 的数据作为验证集
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        data_path,  # 替换为你的数据文件夹路径
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_path,  # 替换为你的数据文件夹路径
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

def build_model(img_width, img_height):
    """构建卷积神经网络模型"""
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator, epochs=10):
    """训练模型并返回训练历史"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[early_stopping]
    )

    return history

def plot_training_history(history):
    """绘制训练和验证准确性与损失"""
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

# 主函数
def main(data_path, img_width, img_height, batch_size, epochs):
    train_generator, validation_generator = create_image_generators(data_path, img_width, img_height, batch_size)
    model = build_model(img_width, img_height)
    history = train_model(model, train_generator, validation_generator, epochs)
    plot_training_history(history)

# 运行主函数，替换 'path_to_data' 为你的数据文件夹路径
main(r'F:\pythonproject\fire_detection1\Fire-Detection-Image-Dataset\Images', img_width, img_height, batch_size, epochs=10)
