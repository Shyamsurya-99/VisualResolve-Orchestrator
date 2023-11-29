import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.utils import draw_bounding_boxes
from torchvision.models import resnet50, ResNet50_Weights
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Object Detection

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
    model.eval()
    return model

model_object_detection = load_object_detection_model()

def make_object_detection_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model_object_detection(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor,
        boxes=prediction["boxes"],
        labels=prediction["labels"],
        colors=["red" if label == "person" else "green" for label in prediction["labels"]],
        width=2,
    )
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

# Image Classification

preprocess_func = ResNet50_Weights.IMAGENET1K_V2.transforms()
categories = np.array(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])

@st.cache_resource
def load_image_classification_model():
    model = resnet50(pretrained=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model

model_image_classification = load_image_classification_model()

def make_image_classification_prediction(model, processed_img):
    probs = model(processed_img.unsqueeze(0))
    probs = probs.softmax(1)
    probs = probs[0].detach().numpy()
    prob, idxs = probs[probs.argsort()[-5:][::-1]], probs.argsort()[-5:][::-1]
    return prob, idxs

def interpret_image_classification_prediction(model, processed_img, target):
    interp_algo = IntegratedGradients(model)
    feature_imp = interp_algo.attribute(processed_img.unsqueeze(0), target=int(target))
    feature_imp = feature_imp[0].numpy()
    feature_imp = feature_imp.transpose(1, 2, 0)
    return feature_imp

# Streamlit Dashboard

st.title("🚀 Explore Object Detection and Image Classification")

task = st.selectbox("Select a task:", ["Object Detection", "Image Classification"])

if task == "Object Detection":
    st.header("Object Detection")
    upload_object_detection = st.file_uploader(
        label="Upload an image for object detection:", type=["png", "jpg", "jpeg"]
    )

    if upload_object_detection:
        img_object_detection = Image.open(upload_object_detection)
        prediction_object_detection = make_object_detection_prediction(img_object_detection)
        img_with_bbox = create_image_with_bboxes(
            np.array(img_object_detection).transpose(2, 0, 1), prediction_object_detection
        )
        fig_object_detection = plt.figure(figsize=(12, 12))
        ax_object_detection = fig_object_detection.add_subplot(111)
        plt.imshow(img_with_bbox)
        plt.xticks([], [])
        plt.yticks([], [])
        ax_object_detection.spines[["top", "bottom", "right", "right"]].set_visible(False)
        st.pyplot(fig_object_detection, use_container_width=True)
        del prediction_object_detection["boxes"]
        st.header("Object Detection Results")
        st.write("Detected objects and their labels:")
        st.write(prediction_object_detection)

elif task == "Image Classification":
    st.header("Image Classification")
    upload_image_classification = st.file_uploader(
        label="Upload an image for image classification:", type=["png", "jpg", "jpeg"]
    )

    if upload_image_classification:
        img_image_classification = Image.open(upload_image_classification)
        preprocessed_img_image_classification = preprocess_func(img_image_classification)
        probs_image_classification, idxs_image_classification = make_image_classification_prediction(
            model_image_classification, preprocessed_img_image_classification
        )
        feature_img_image_classification = interpret_image_classification_prediction(
            model_image_classification, preprocessed_img_image_classification, idxs_image_classification[0]
        )

        main_fig_image_classification = plt.figure(figsize=(12, 3))
        ax_main_image_classification = main_fig_image_classification.add_subplot(111)
        plt.barh(y=categories[idxs_image_classification], width=probs_image_classification, color="tomato")
        plt.title("Top 5 Probabilities", loc="center", fontsize=15)
        st.pyplot(main_fig_image_classification, use_container_width=True)

        interp_fig_image_classification, ax_interp_image_classification = viz.visualize_image_attr(
            feature_img_image_classification, show_colorbar=True, fig_size=(6, 6)
        )

        col1_image_classification, col2_image_classification = st.columns(2, gap="medium")
        with col1_image_classification:
            main_fig_image_classification = plt.figure(figsize=(6, 6))
            ax_main_image_classification = main_fig_image_classification.add_subplot(111)
            plt.imshow(img_image_classification)
            plt.xticks([], [])
            plt.yticks([], [])
            st.pyplot(main_fig_image_classification, use_container_width=True)

        with col2_image_classification:
            st.pyplot(interp_fig_image_classification, use_container_width=True)
