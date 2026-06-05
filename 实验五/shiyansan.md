```python
import tarfile
from pathlib import Path

import numpy as np
import tensorflow as tf

# TensorFlow 官方花卉数据集。第一次运行时会自动下载，之后会复用本地缓存。
FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

print("TensorFlow 版本:", tf.__version__)

```

    TensorFlow 版本: 2.21.0
    


```python
# 数据目录配置：
# - DATA_DIR = None：自动下载并使用 TensorFlow 官方 flowers 数据集。
# - DATA_DIR = r"D:\path\to\my_images"：使用你自己的图片分类目录。
#
# 自定义图片目录需要按类别分文件夹，例如：
# my_images/
#   daisy/
#     1.jpg
#   roses/
#     2.jpg
DATA_DIR = None

# 导出目录。训练完成后会在这里生成 model.tflite、labels.txt 和 flower_classifier.keras。
EXPORT_DIR = "exported_flower_model"

# 训练参数。教程演示可以先用 3 到 5 个 epoch；如果使用自己的数据，可以适当增加。
EPOCHS = 5
BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 1e-3

# TFLite 量化方式：
# - "dynamic"：默认推荐，模型更小，通常最容易成功。
# - "float16"：适合部分支持 float16 的设备。
# - "int8"：体积更小，但需要代表性数据集，转换要求更严格。
# - "none"：不量化，保留浮点模型。
QUANTIZATION = "dynamic"

# 固定随机种子，方便训练/验证划分尽量可复现。
SEED = 123

```


```python
def load_flower_datasets(data_dir, image_size, batch_size, seed):
    # 如果没有传入自定义数据目录，就下载 TensorFlow 官方 flower_photos 数据集。
    if data_dir is None:
        archive_path = tf.keras.utils.get_file(
            "flower_photos.tgz",
            FLOWER_URL,
            extract=False,
        )
        archive_path = Path(archive_path)

        # Keras 可能已经缓存了解压后的目录；先检查常见位置，避免重复解压。
        candidates = [
            archive_path.parent / "flower_photos",
            archive_path.parent / "flower_photos_extracted" / "flower_photos",
        ]
        data_dir = next((path for path in candidates if path.exists()), None)
        if data_dir is None:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(archive_path.parent / "flower_photos_extracted")
            data_dir = archive_path.parent / "flower_photos_extracted" / "flower_photos"
    else:
        data_dir = Path(data_dir)

    # 从目录读取图片。目录下的每个子文件夹会被当作一个类别。
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )
    class_names = train_ds.class_names

    # 原始 validation 部分再拆成验证集和测试集：验证集用于训练过程中观察效果，测试集用于最后评估。
    val_batches = int(tf.data.experimental.cardinality(val_ds).numpy())
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)

    # cache/prefetch 可以减少数据读取等待；shuffle 只用于训练集。
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names

```


```python
train_ds, val_ds, test_ds, class_names = load_flower_datasets(
    DATA_DIR,IMAGE_SIZE,BATCH_SIZE,SEED
)
print("类别数量:", len(class_names))
print("类别名称:", class_names)
```

    Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    [1m228813984/228813984[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m19s[0m 0us/step
    Found 3670 files belonging to 5 classes.
    Using 2936 files for training.
    Found 3670 files belonging to 5 classes.
    Using 734 files for validation.
    类别数量: 5
    类别名称: ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    


```python
def build_model(num_classes, image_size, learning_rate):
    # 输入图片尺寸固定为 IMAGE_SIZE x IMAGE_SIZE x 3。
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")

    # MobileNetV2 有自己的预处理方式，这里把像素值转换到模型期望的范围。
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # include_top=False 表示不要 ImageNet 原始的 1000 类分类头，只保留特征提取部分。
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    # 冻结预训练模型参数，只训练后面的 Dense 分类层。
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)

    # 输出维度等于类别数量，softmax 输出每个类别的概率。
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

```


```python
# 创建模型并打印结构。第一次运行会下载 MobileNetV2 的 ImageNet 预训练权重。
model = build_model(len(class_names), IMAGE_SIZE, LEARNING_RATE)
model.summary()

```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
    [1m9406464/9406464[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 0us/step
    WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ image (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ true_divide (<span style="color: #0087ff; text-decoration-color: #0087ff">TrueDivide</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ subtract (<span style="color: #0087ff; text-decoration-color: #0087ff">Subtract</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ mobilenetv2_1.00_224            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1280</span>)           │     <span style="color: #00af00; text-decoration-color: #00af00">2,257,984</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)                    │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1280</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ predictions (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>)              │         <span style="color: #00af00; text-decoration-color: #00af00">6,405</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,264,389</span> (8.64 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,405</span> (25.02 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,257,984</span> (8.61 MB)
</pre>




```python
# 开始训练。history 中会保存每个 epoch 的 loss、accuracy、val_loss、val_accuracy。
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

```

    Epoch 1/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 187ms/step - accuracy: 0.6809 - loss: 0.8489 - val_accuracy: 0.8796 - val_loss: 0.4052
    Epoch 2/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m16s[0m 174ms/step - accuracy: 0.8491 - loss: 0.4213 - val_accuracy: 0.8927 - val_loss: 0.3414
    Epoch 3/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m16s[0m 175ms/step - accuracy: 0.8822 - loss: 0.3406 - val_accuracy: 0.9031 - val_loss: 0.2941
    Epoch 4/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m16s[0m 176ms/step - accuracy: 0.8971 - loss: 0.2976 - val_accuracy: 0.9162 - val_loss: 0.2676
    Epoch 5/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m16s[0m 179ms/step - accuracy: 0.9186 - loss: 0.2519 - val_accuracy: 0.9136 - val_loss: 0.2552
    


```python
# 使用测试集评估模型。测试集没有参与训练，用于更客观地观察最终效果。
loss, accuracy = model.evaluate(test_ds)
print(f"test_loss={loss:.4f}, test_accuracy={accuracy:.4f}")

```

    [1m11/11[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 173ms/step - accuracy: 0.8864 - loss: 0.3090
    test_loss=0.3090, test_accuracy=0.8864
    


```python
def convert_to_tflite(model, quantization, representative_ds):
    # 从 Keras 模型创建 TFLite 转换器。
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "dynamic":
        # 动态范围量化：最常用、最容易成功的压缩方式。
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        # float16 量化：权重使用半精度浮点数，适合部分移动端/GPU 场景。
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        # int8 全整数量化：体积更小，但需要代表性数据集校准输入分布。
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_data_gen():
            for images, _ in representative_ds.take(100):
                for image in images:
                    yield [tf.expand_dims(tf.cast(image, tf.float32), 0)]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quantization != "none":
        raise ValueError(f"Unsupported quantization mode: {quantization}")

    return converter.convert()

```


```python
# 创建导出目录。
export_dir = Path(EXPORT_DIR)
export_dir.mkdir(parents=True, exist_ok=True)

# 保存标签文件。部署时需要 labels.txt 把模型输出编号映射回类别名称。
labels_path = export_dir / "labels.txt"
labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

# 保存 Keras 原始模型，便于以后继续训练或重新转换。
keras_path = export_dir / "flower_classifier.keras"
model.save(keras_path)

# 转换并保存 TFLite 模型。
tflite_model = convert_to_tflite(model, QUANTIZATION, train_ds)
tflite_path = export_dir / "model.tflite"
tflite_path.write_bytes(tflite_model)

print(f"已保存 Keras 模型: {keras_path}")
print(f"已保存 TFLite 模型: {tflite_path}")
print(f"已保存标签文件: {labels_path}")

```

    INFO:tensorflow:Assets written to: C:\Users\DELL\AppData\Local\Temp\tmpd553juj_\assets
    

    INFO:tensorflow:Assets written to: C:\Users\DELL\AppData\Local\Temp\tmpd553juj_\assets
    

    Saved artifact at 'C:\Users\DELL\AppData\Local\Temp\tmpd553juj_'. The following endpoints are available:
    
    * Endpoint 'serve'
      args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='image')
    Output Type:
      TensorSpec(shape=(None, 5), dtype=tf.float32, name=None)
    Captures:
      2865231147632: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231149392: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231148160: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231147456: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231148864: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865224509824: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223520272: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223522736: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223518512: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223521504: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223525376: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223528192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223530480: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223526432: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223529248: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223530304: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223585104: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223586160: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223582640: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223585280: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865224506480: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223592672: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223595312: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223590912: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223593200: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223595136: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223634960: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223636016: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223632496: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223635136: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223637248: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223640064: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223642352: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223638304: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223641120: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223636192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223713008: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223713184: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223714592: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223714768: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223718464: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223721280: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223723568: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223719520: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223722336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223726736: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223763040: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223727792: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223728320: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223761984: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223645168: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223770784: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223773424: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223769024: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223771312: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865223776768: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231153104: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231151344: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231153456: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231153280: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231155392: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231158208: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231160496: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231156448: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231159264: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231166656: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231162608: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231164544: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231165952: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231166832: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865230728704: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865230731520: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865230733808: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865230729760: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865230732576: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865230736448: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865230739264: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865230738032: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865230737504: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865230740320: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231163312: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231238368: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231241008: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231236608: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231238896: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231243648: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231246464: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231248752: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231244704: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231247520: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231248576: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231008640: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231010928: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231006880: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231009696: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231236080: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231017616: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231016384: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231015856: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231018144: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231612912: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231615728: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231618016: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231613968: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231616784: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231619248: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231622064: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231624352: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231620304: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231623120: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231015328: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231727952: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231730592: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231726192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231728480: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231733232: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231736048: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231738336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231734288: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231737104: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231738160: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865204940992: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865204942048: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865204938528: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865204941168: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865231725664: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865204947152: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865204949792: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865204945392: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865204947680: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865204949616: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205054976: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205056032: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205052512: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205055152: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205058672: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205061488: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205063776: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205059728: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205062544: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205056208: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205134432: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205137248: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205135664: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205136192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205139888: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205142704: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205144992: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205140944: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205143760: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205148336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205250176: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205250704: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205250528: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205250352: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205066592: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205256336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205258976: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205254576: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205256864: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205262320: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205413312: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205412080: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205261616: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205413488: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205415600: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205418416: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205420704: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205416656: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205419472: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205426864: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205425808: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205424576: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205424048: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205426336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205513200: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205516016: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205518304: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205514256: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205517072: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205520944: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205523760: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205526048: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205522000: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205524816: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205423520: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205613264: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205615904: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205611504: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205613792: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205618544: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205621360: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205623648: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205619600: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205622416: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205623472: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205694832: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205697120: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205693072: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205695888: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205610976: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205703808: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205706448: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205702048: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205704336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205706272: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205794016: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205796304: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205792256: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205795072: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205797536: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205800352: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205802640: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205798592: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205801408: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865205796480: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222503056: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222505872: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222502000: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222504816: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222508512: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222511328: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222513616: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222509568: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222512384: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222513440: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222586032: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222587088: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222583568: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222586208: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222500944: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222592192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222594832: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222590432: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222592720: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222598176: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222700016: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222697024: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222697552: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222700192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222703712: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222706528: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222708816: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222704768: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222707584: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222701248: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222763088: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222765376: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222765200: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222764848: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222768544: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222771360: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222773648: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222769600: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222772416: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222774880: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222774352: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222776640: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222778048: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222860864: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222872832: TensorSpec(shape=(), dtype=tf.resource, name=None)
      2865222870896: TensorSpec(shape=(), dtype=tf.resource, name=None)
    已保存 Keras 模型: exported_flower_model\flower_classifier.keras
    已保存 TFLite 模型: exported_flower_model\model.tflite
    已保存标签文件: exported_flower_model\labels.txt
    


```python
def smoke_test_tflite(tflite_path, test_ds, class_names):
    # 加载 TFLite 模型并分配张量内存。
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 从测试集中取 8 张图片做快速推理。
    images, labels = next(iter(test_ds.unbatch().batch(8)))
    input_data = tf.cast(images, input_details["dtype"]).numpy()

    # 如果模型是 uint8 输入，需要按照量化参数把图片转换到对应范围。
    if input_details["dtype"] == np.uint8:
        scale, zero_point = input_details["quantization"]
        if scale:
            input_data = images.numpy() / scale + zero_point
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)

    predictions = []
    for image in input_data:
        interpreter.set_tensor(input_details["index"], np.expand_dims(image, 0))
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_details["index"])[0])

    predicted_ids = np.argmax(np.asarray(predictions), axis=1)
    for expected, predicted in zip(labels.numpy()[:5], predicted_ids[:5]):
        print(f"真实类别={class_names[expected]}, 预测类别={class_names[predicted]}")

```


```python
# 运行 TFLite 快速测试。
smoke_test_tflite(tflite_path, test_ds, class_names)

```

    c:\Users\DELL\anaconda3\envs\tf10\lib\site-packages\tensorflow\lite\python\interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
        TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
        See the [migration guide](https://ai.google.dev/edge/litert/migration)
        for details.
        
      warnings.warn(_INTERPRETER_DELETION_WARNING)
    

    真实类别=daisy, 预测类别=daisy
    真实类别=tulips, 预测类别=tulips
    真实类别=sunflowers, 预测类别=sunflowers
    真实类别=daisy, 预测类别=daisy
    真实类别=sunflowers, 预测类别=sunflowers
    
