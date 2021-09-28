# openvino
Repository for examples on how to use OpenVINO

Note that models/ does not contain the original models, but only the converted models. This is because the original models' size exceeded GitHub's maximum file size of 50.00 MB. To get the original models use the OpenVINO downloader tool and download the models again.

models/ssd_mobilenet_v2_coco/ only contains the FP16 precision and not the FP32 precision which is also included in the download from OpenVINO since the file size of the .bin file exceedes GitHub's maximum file size of 50.00 MB. It's possible that bettery accuracy can be achieved by downloading the model and using the FP32 precision instead.