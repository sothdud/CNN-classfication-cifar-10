
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import time

TEST_DATA_DIR = r'/app/test'
MODEL_SAVE_PATH = r'/app/saved_model/cifar10_model100.keras'
OUTPUT_DIR = r'/app/out_put'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("test dataset load")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print("저장된 모델 로딩...")
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

print("테스트셋 평가 중...")
loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")


#추론 시간 출력
start_time = time.time()
predictions = model.predict(test_ds, verbose=1)  # 전체 test_ds 예측
end_time = time.time()

total_inference_time = end_time - start_time
print(f"전체 추론 시간: {total_inference_time:.4f} 초")


num_images = len(list(test_ds.unbatch()))
avg_inference_time = total_inference_time / num_images
print(f"평균 이미지당 추론 시간: {avg_inference_time:.6f} 초")


#시각화
for images, labels in test_ds.take(2):  #take(1)->배치 1개만
    for i in range(5):
        image = images[i]
        true_label_index = labels[i].numpy()
        true_label_name = class_names[true_label_index]

        # 예측을 위해 이미지 차원을 (32, 32, 3) -> (1, 32, 32, 3)으로 확장
        img_array = tf.expand_dims(image, 0)



        #예측
        predictions = model.predict(img_array)
        score = predictions[0]


        #가장 높은 확률을 가진 클래스의 인덱스와 이름
        predicted_label_index = np.argmax(score)
        predicted_label_name = class_names[predicted_label_index]

        #시각화
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image.numpy().astype("uint8"))
        #예측 결과 맞으면 초록/틀리면 빨강
        title_color = "green" if predicted_label_index == true_label_index else "red"
        plt.title(f"True: {true_label_name}\nPredicted: {predicted_label_name} ({100 * np.max(score):.2f}%)", color=title_color)
        plt.axis("off")

        #클래스별 예측 그래프
        plt.subplot(1, 2, 2)
        bars = plt.barh(class_names, score)
        plt.xlabel("Probability")
        plt.title("Class Probabilities")
        plt.xlim([0, 1]) # 확률이므로 x축 범위를 0에서 1로 고정

        #예측 정답클래스 막대 색
        bars[true_label_index].set_color('gray')
        bars[predicted_label_index].set_color('C0' if title_color == "green" else 'red')


        #저장
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'prediction_result_{i}.png'))
        plt.show()


