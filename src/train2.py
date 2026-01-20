
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

    # ==================================================================
    # 🚀 1단계: 특성 추출 (Feature Extraction)
    # ==================================================================
    print("--- 1단계: 특성 추출(몸통은 고정) 시작 ---")
    model = efficientNetB0()  # model.py에서 ef_base.trainable = False 상태

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # 1단계 EarlyStopping: 비교적 여유롭게 설정
    early_stopping_1 = EarlyStopping(
        monitor='val_loss',
        patience=10,  # 충분한 학습을 위해 patience 증가
        verbose=1,
        restore_best_weights=True
    )

    history = model.fit(train_ds,
                        epochs=50,  # 어차피 조기 종료됨
                        validation_data=val_ds,
                        callbacks=[early_stopping_1])

    print("\n--- 1단계 완료. 최적 val_accuracy: {:.4f} ---".format(max(history.history['val_accuracy'])))

    # ==================================================================
    # 🚀 2단계: 미세 조정 (Fine-Tuning)
    # ==================================================================
    print("\n--- 2단계: 미세 조정(몸통 전체 학습) 시작 ---")

    # 1. Base 모델의 동결을 해제합니다.
    # model.summary()를 보면 ef_base는 두 번째 레이어입니다. (첫 번째는 Resizing)
    model.layers[1].trainable = True
    print("EfficientNetB0 Base 모델의 동결을 해제했습니다.")

    # 2. ❗❗❗ 아주 낮은 학습률로 다시 컴파일합니다. (가장 중요) ❗❗❗
    model.compile(optimizer=Adam(learning_rate=1e-5),  # 0.00001
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    print("매우 낮은 학습률(1e-5)로 모델을 다시 컴파일했습니다.")
    model.summary()

    # 2단계 EarlyStopping: 더 민감하게 설정하여 과적합 방지
    early_stopping_2 = EarlyStopping(
        monitor='val_loss',
        patience=5,  # 과적합이 시작되면 빠르게 중단
        verbose=1,
        restore_best_weights=True
    )

    # 3. 추가 학습을 진행합니다.
    # 이전 학습이 끝난 epoch에서 이어서 학습합니다.
    fine_tune_epochs = 30  # 추가 학습 에폭
    total_epochs = len(history.epoch) + fine_tune_epochs

    history_fine = model.fit(train_ds,
                             epochs=total_epochs,
                             initial_epoch=len(history.epoch),  # 이전 학습 지점부터 시작
                             validation_data=val_ds,
                             callbacks=[early_stopping_2])

    model.save(MODEL_SAVE_PATH)
    plot_and_save(history_fine, OUTPUT_DIR)
    print("\n--- 모든 학습 완료. 최종 모델이 {}에 저장되었습니다. ---".format(MODEL_SAVE_PATH))
