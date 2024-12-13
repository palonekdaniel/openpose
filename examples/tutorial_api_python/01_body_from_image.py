# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import json
from sys import platform
import argparse
import pyopenpose as op

try:
    """
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and t>
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
    """

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../examples/tutorial_api_python/input_front_correct.jpg", help="Process an image. Read all standard formats (jpg, png, bmp>")
    parser.add_argument("--output-json", default="keypoints.json", help="Output json file")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/openpose/models/"
    params["net_resolution"] = "320x176"
    params["model_pose"] = "BODY_25"

    #params["num_gpu"] = 0                # Disable GPU usage
    #params["num_gpu_start"] = 0
    #params["disable_multi_thread"] = True  # Ensure only one thread is usedwq


    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    keypoints = datum.poseKeypoints.tolist() if datum.poseKeypoints is not None else None
    output_json_path = args[0].output_json
    with open(output_json_path, 'w') as f:
        json.dump({"keypoints": keypoints}, f, indent=4)
    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imwrite("result_body.jpg", datum.cvOutputData)
    #cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
    #cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)
