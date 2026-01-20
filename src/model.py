from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0

#6개
def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        #입력이미지 픽셀 값(0~255)을 0~1 사이 값으로 졍규화

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # Batch Normalization: 각 배치의 출력을 정규화하여 학습을 안정화하고 속도를 높입니다.
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        #Dense(Fully-connected) Layer: 추출된 모든 특징을 조합하여 클래스를 예측하는 역할
        layers.Dense(256, activation='relu'),
        #dropout: 훈련 시 뉴런의 50%를 무작위로 비활성화하여 과대적합(overfitting)을 방지
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


#8개 레이어
def build_cnn2(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        layers.Rescaling(1. / 255, input_shape=input_shape),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')

    ])
    return model


#leakyrelu 사용 / 8개 layer
def build_cnn5(input_shape=(32, 32, 3), num_classes=10, alpha=0.01):
    model = models.Sequential([
        layers.Rescaling(1. / 255, input_shape=input_shape),


        layers.Conv2D(32, (3, 3), padding='same'), # 'activation' 제거
        layers.LeakyReLU(alpha=alpha),            # Leaky ReLU 추가
        layers.Conv2D(32, (3, 3), padding='same'), # 'activation' 제거
        layers.LeakyReLU(alpha=alpha),            # Leaky ReLU 추가
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),


        layers.Conv2D(64, (3, 3), padding='same'), # 'activation' 제거
        layers.LeakyReLU(alpha=alpha),            # Leaky ReLU 추가
        layers.Conv2D(64, (3, 3), padding='same'), # 'activation' 제거
        layers.LeakyReLU(alpha=alpha),            # Leaky ReLU 추가
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),


        layers.Conv2D(128, (3, 3), padding='same'), # 'activation' 제거
        layers.LeakyReLU(alpha=alpha),             # Leaky ReLU 추가
        layers.Conv2D(128, (3, 3), padding='same'), # 'activation' 제거
        layers.LeakyReLU(alpha=alpha),             # Leaky ReLU 추가
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),


        layers.Conv2D(256, (3, 3), padding='same'), # 'activation' 제거
        layers.LeakyReLU(alpha=alpha),             # Leaky ReLU 추가
        layers.Conv2D(256, (3, 3), padding='same'), # 'activation' 제거
        layers.LeakyReLU(alpha=alpha),             # Leaky ReLU 추가
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(256),
        layers.LeakyReLU(alpha=alpha),             # Leaky ReLU 추가
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax') # 최종 출력층은 softmax 유지
    ])
    return model


model_leaky_relu = build_cnn5()
model_leaky_relu.summary()

#10개 레이어
def build_cnn4(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        layers.Rescaling(1. / 255, input_shape=input_shape),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')

    ])
    model.summary()
    return model




#bathnormalization 위치. 배치 노말리제이션이 활성화 함수에 들어가는 입력값자체를 정규화해줌.. 애매.. validation이 너무낮음.
def build_cnn3(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        layers.Rescaling(1. / 255, input_shape=input_shape),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')

    ])
    return model




#https://www.kaggle.com/code/shivamb/cnn-architectures-vgg-resnet-inception-tl#1.4-Resnets
def vgg19(num_classes=10):
    vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    vgg19_base.trainable = False

    model = models.Sequential([
        vgg19_base,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    print("VGG19 전이 학습 모델 생성")
    model.summary()
    return model




def resnet50(num_classes=10):
    resnet9_base =ResNet50(weights='imagenet', include_top=False)
    resnet9_base.trainable = False

    model = models.Sequential([
        layers.Resizing(224, 224, input_shape=(32, 32, 3)),
        resnet9_base,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    print("resnet50 전이 학습 모델 생성")
    model.summary()
    return model



def efficientNetB0(num_classes=10):
    ef_base =EfficientNetB0(include_top=False,weights='imagenet', classes=num_classes,
                                                             input_shape=(224, 224, 3))

    ef_base.trainable = True
    for layer in ef_base.layers[:150]: #150까지는 학습 못하게 freeze, 그 이후는 학습할수잇도록함.
        layer.trainable = False
    #true 가중치를 다시 학습시킴.

    model = models.Sequential([

        ef_base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    print("efficientNetB0 전이 학습 모델 생성")
    model.summary()
    return model
