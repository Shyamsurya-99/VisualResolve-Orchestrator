# VisualResolve-Orchestrator(object detection and image classification)


## Overview

The Visual Decision Engine is a web application that empowers users to choose between object detection and image classification tasks. Built with Streamlit, this project seamlessly integrates state-of-the-art computer vision models. Users can upload images, make selections, and interactively visualize results.

## Features

- Choose between Object Detection and Image Classification tasks.
- Upload images for analysis.
- View real-time results with bounding boxes for Object Detection.
- Explore top probability classes and interpretability for Image Classification.

## Prerequisites

- Python 3.6+
- Install required dependencies by running: `pip install -r requirements.txt`

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/visual-decision-engine.git
    cd visual-decision-engine
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application:**

    ```bash
    streamlit run app.py
    ```

2. **Open the provided URL in your web browser.**

## Demo

![Demo](https://github.com/Shyamsurya-99/VisualResolve-Orchestrator/assets/110275462/23d6cdc4-21af-436e-b72f-4823e7e8fef2)


Include a GIF or screenshot of your application in action.

## Model Information

- **Object Detection Model:** Faster R-CNN with ResNet50 backbone.
- **Image Classification Model:** ResNet50 with pre-trained weights.

## Acknowledgments

- This project uses [Streamlit](https://streamlit.io/) for the web interface.
- Models are from [torchvision](https://pytorch.org/vision/stable/index.html).
- [Captum](https://captum.ai/) is used for interpretability.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact

- **Your Name:** Shyam surya G
- **Email:** shyamsurya54@gmail.com
