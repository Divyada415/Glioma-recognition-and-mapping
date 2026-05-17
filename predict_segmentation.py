import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

IMG_SIZE = 256

model = load_model("unet_model.h5")

img_path ="glioma_project/test_images/TCGA_CS_4941_19960909_12.tif"
mask_path ="glioma_project/test_masks/TCGA_CS_4941_19960909_12_mask.tif"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if img is None or true_mask is None:
    print("ERROR: Image or Mask not found")
    exit()

img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
true_mask = cv2.resize(true_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
true_mask = (true_mask > 0.05).astype(np.uint8)

img_norm = img_resized / 255.0
img_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

pred = model.predict(img_input)[0]
pred_mask = (pred > 0.1).astype(np.uint8)
pred_mask = np.squeeze(pred_mask)

intersection = np.sum(pred_mask * true_mask)
sum_pred = np.sum(pred_mask)
sum_true = np.sum(true_mask)

if sum_true == 0 and sum_pred == 0:
    dice = 1.0
elif sum_true == 0 or sum_pred == 0:
    dice = 0.0
else:
    dice = (2.0 * intersection) / (sum_pred + sum_true)

print("Dice Score:", dice)

cv2.imwrite("predicted_mask.png", pred_mask * 255)
cv2.imwrite("true_mask.png", true_mask * 255)

overlay = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
overlay[pred_mask == 1] = [0, 0, 255]
cv2.imwrite("overlay.png", overlay)



if dice > 0.8 and np.sum(pred_mask) > 0:

    print("Applying Explainable AI (Grad-CAM)...")

    
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    gradcam_overlay = cv2.addWeighted(
        cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR),
        0.6,
        heatmap_color,
        0.4,
        0
    )

    cv2.imwrite("gradcam_heatmap.png", heatmap_color)
    cv2.imwrite("gradcam_overlay.png", gradcam_overlay)

    print("Grad-CAM generated successfully.")
    print("gradcam_heatmap.png")
    print("gradcam_overlay.png")

    print("\nTHEORETICAL EXPLANATION:")
    print("Grad-CAM highlights regions that influenced tumor prediction.")
    print("The highlighted region corresponds to the detected glioma area.")

else:
    print("Explainable AI not applied (Dice < 0.8 or no tumor detected)")