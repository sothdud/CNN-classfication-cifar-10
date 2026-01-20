import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np


# TEST_DATA_DIR = r'/app/data/cifar-10_test' # 더 이상 필요 없음
MODEL_SAVE_PATH = r'/app/saved_model/cifar10_model.keras'
OUTPUT_DIR = r'/app/out_put'
IMG_SIZE = (32, 32)
BATCH_SIZE = 32

print("TensorFlow CIFAR-10 testdataset load...")
(_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
# 라벨 차원 축소 (10000, 1) -> (10000,)
test_ds = test_ds.map(lambda image, label: (image, tf.squeeze(label)))
# 데이터 배치 단위 묶음
test_ds = test_ds.batch(BATCH_SIZE)
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 변경된 부분 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print("저장된 모델 로딩...")
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

print("테스트셋 평가 중...")
loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")


# 시각화
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("\n개별 이미지 예측 결과 시각화...")
for images, labels in test_ds.take(1):
    for i in range(5):
        image = images[i]
        true_label_index = labels[i].numpy()
        true_label_name = class_names[true_label_index]

        img_array = tf.expand_dims(image, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_label_index = np.argmax(score)
        predicted_label_name = class_names[predicted_label_index]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image.numpy().astype("uint8"))
        title_color = "green" if predicted_label_index == true_label_index else "red"
        plt.title(f"True: {true_label_name}\nPredicted: {predicted_label_name} ({100 * np.max(score):.2f}%)", color=title_color)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        bars = plt.barh(class_names, score)
        plt.xlabel("Probability")
        plt.title("Class Probabilities")
        plt.xlim([0, 1])

        bars[true_label_index].set_color('gray')
        bars[predicted_label_index].set_color('C0' if title_color == "green" else 'red')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'prediction_result_{i}.png'))
        plt.show()