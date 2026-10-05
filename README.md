# MobileSAM to ONNX

English | [日本語](README.ja.md)

A Python script to convert (export) MobileSAM (Mobile Segment Anything Model) pretrained weights (`.pt`) into **two ONNX files (Image Encoder and Mask Decoder)** for cross-platform, high-performance inference.

---

## Verified Environment

This script and the exported ONNX models have been verified in the following version environment.

| Package / Tool                  | Version                      |
| :------------------------------ | :--------------------------- |
| **Python**                      | `3.10.11`                    |
| **PyTorch** (`torch`)           | `2.0.1`                      |
| **TorchVision** (`torchvision`) | `0.15.2`                     |
| **ONNX** (`onnx`)               | `1.13.0`                     |
| **NumPy** (`numpy`)             | `<2` (Less than version 2.0) |

---

## Setup

1. **Place the MobileSAM Package**  
   To run this script, the official `MobileSAM/mobile_sam` directory (package) must exist within your project. Clone or download the source code from the [Official MobileSAM Repository](https://github.com/chaoningzhang/mobilesam) and place it in your working directory.

2. **Place the Pretrained Weights**  
   Obtain the MobileSAM weight file (`mobile_sam.pt`) from the official repository and place it according to the following directory structure:
    ```text
    project_root/
    ├── MobileSAM/                # Official repository directory
    │   ├── mobile_sam/
    │   ├── weights/
    │   │   └── mobile_sam.pt     # Pretrained weights file
    │   └── LICENSE
    ├── export_mobile_sam_onnx.py # This export script
    └── README.md
    ```

---

## Constants & Configuration Parameters

The details of the configuration values defined at the beginning of the script are as follows. They can be updated collectively according to your operational environment or model specifications.

| Constant Name              | Default Value                         | Description                                                |
| :------------------------- | :------------------------------------ | :--------------------------------------------------------- |
| `CHECK_POINT_FILE_PATH`    | `"./MobileSAM/weights/mobile_sam.pt"` | Path to the input pretrained weights (checkpoint) file     |
| `ENCODER_OUTPUT_FILE_PATH` | `"./mobile_sam_image_encoder.onnx"`   | Path for the exported image encoder ONNX file              |
| `DECODER_OUTPUT_FILE_PATH` | `"./mobile_sam_mask_decoder.onnx"`    | Path for the exported mask decoder ONNX file               |
| `DEVICE`                   | `"cpu"`                               | Device used to load and export the model (`cpu` or `cuda`) |
| `MODEL_TYPE`               | `"vit_t"`                             | MobileSAM model architecture type                          |
| `RESOLUTION_WIDTH`         | `1024`                                | Standard width of the input image (pixels)                 |
| `RESOLUTION_HEIGHT`        | `1024`                                | Standard height of the input image (pixels)                |
| `OPSET_VERSION`            | `17`                                  | Opset version for ONNX export                              |
| `SINGLE_MASK`              | `True`                                | Flag to return only a single mask in the decoder output    |

---

## Processing Flow

The script calls the `export_mobile_sam` function upon execution and performs the process in three steps:

### 1. Model Loading

- Instantiates the MobileSAM model matching the `MODEL_TYPE` (`vit_t`) from the specified `CHECK_POINT_FILE_PATH`.
- Places the model on the specified `DEVICE` and switches to evaluation mode via the `eval()` method.

### 2. Exporting the Image Encoder

- **Purpose:** Extracts the component that computes feature maps (`image_embeddings`) from the input image.
- **Dummy Input:** Creates a random floating-point tensor with a shape of `(1, 3, RESOLUTION_HEIGHT, RESOLUTION_WIDTH)`.
- **Output:** Writes out an ONNX model that outputs `image_embeddings` to the specified path.

### 3. Exporting the Mask Decoder

- **Purpose:** Extracts the component that predicts the final segmentation mask from image features and prompts (such as points or bounding boxes).
- **Wrapper:** Wraps using the official `SamOnnxModel` with the `SINGLE_MASK` configuration applied.
- **Dummy Inputs:**
    - Image features (`image_embeddings`)
    - Prompt coordinates (`point_coords`)
    - Prompt labels (`point_labels`)
    - Mask input (`mask_input`)
    - Has mask input flag (`has_mask_input`)
    - Original image size (`orig_im_size`)
- **Dynamic Axes:**
    - Configured so that the number of prompt points (`num_points`) can be handled dynamically.
- **Output:** Writes out an ONNX model that outputs `masks` (predicted masks) and `iou_predictions` (IoU confidence scores).

---

## Usage

1. Place the weight file at the designated path (`./MobileSAM/weights/mobile_sam.pt`).
2. Run the script:

```bash
python export_mobile_sam_onnx.py
```

---

## Disclaimer

The author assumes no responsibility for any damages, troubles, or adverse effects caused by the use of this script. Please use it entirely at your own risk.

---

## License

The script files in this repository are published under the [MIT License](LICENSE).

However, the code and files under the `MobileSAM/` directory are based on the original [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM) repository (Apache License 2.0). Please refer to `Mobile`SAM/LICENSE` for details.
