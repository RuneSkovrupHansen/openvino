import cv2
import json

from common import prepare_image
from common import draw_bounding_boxes

from openvino.inference_engine import IECore

"""

Model, ssd_mobilenet_v1_cocomodels/ssd_mobilenet_v1_coco/classes
https://docs.openvinotoolkit.org/latest/omz_models_model_ssd_mobilenet_v1_coco.html

Classes
https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/coco_91cl_bkgr.txt

"""

def main():

    # Specify cv2 window name
    _window_name = "detections"

    # Load model
    model_name = "ssd_mobilenet_v2_coco"
    model_precision = "FP16"
    
    model_xml_path = f"models/{model_name}/{model_precision}/{model_name}.xml"
    model_bin_path = f"models/{model_name}/{model_precision}/{model_name}.bin"

    # Initialize inference engine
    ie = IECore()

    # Read network
    network = ie.read_network(model_xml_path, model_bin_path)

    # Find input shape, layout and size
    input_name = next(iter(network.input_info))
    input_data = network.input_info[input_name].input_data
    input_shape = input_data.shape # [1, 3, 300, 300]
    input_layout = input_data.layout # NCHW
    input_size = (input_shape[2], input_shape[3]) # (300, 300)

    # Load network
    device = "CPU" # MYRIAD / CPU
    exec_network = ie.load_network(network=network, device_name=device, num_requests=1)

    # Load classes
    classes_path = f"models/{model_name}/classes.json"
    with open(classes_path) as f:
        classes = f.read()
    
    classes = json.loads(classes)

    # Check for camera
    camera = cv2.VideoCapture(0)
    if camera is None or not camera.isOpened():
        print("Unable to find camera")
        quit()

    print("Running object detection on camera, press [q] to quit")

    while True:
        ret, frame = camera.read()

        if not ret:
            continue

        frame_prepared = prepare_image(frame, target_size=input_size, target_layout=input_layout)

        output = exec_network.infer({input_name: frame_prepared})
        detections = output["DetectionOutput"]

        cv2.imshow(_window_name, draw_bounding_boxes(frame, detections, classes))
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    camera.release()

if __name__ == "__main__":
    main()                                                                                                                              