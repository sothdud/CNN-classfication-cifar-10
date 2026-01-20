
import tensorflow as tf
import os
from data_loader import load_datasets
from model import build_cnn2
from model import vgg19
from model import efficientNetB0
from model import build_cnn3
from model import build_cnn4
from model import build_cnn5
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from utils import plot_and_save


TRAIN_DIR = r'/app/train'
VAL_DIR = r'/app/validation'
MODEL_SAVE_PATH = r'/app/saved_model/cifar10_model100.keras'
OUTPUT_DIR = r'/app/output'

if __name__ == "__main__":
    if tf.config.list_physical_devices('GPU'):
        print("GPU 활성화 확인됨")
    else:
        print("CPU 사용")

    train_ds, val_ds, class_names = load_datasets(TRAIN_DIR, VAL_DIR)


    #이 모델명만 수정하면됨
    model = efficientNetB0()
    adam_optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=adam_optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])


    early_stopping = EarlyStopping(
        monitor='val_loss',  # 검증 데이터의 손실(loss)을 관찰
        patience=7, # 10 에포크 동안 개선되지 않으면 중단
        min_delta=0.001,
        verbose=1,  # 중단 시 메시지 출력
        restore_best_weights=True  # 가장 성능 좋았던 가중치로 복원
    )


    history = model.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=[early_stopping])
    model.save(MODEL_SAVE_PATH)
    plot_and_save(history, OUTPUT_DIR)


