import base64
import json

import cv2
import numpy as np
import pyopenpose as op
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic.v1 import validator

app = FastAPI()

class ImageProcessingRequest(BaseModel):
    front_img: str
    profile_img: str

    @validator('front_img')
    def validate_person_height(cls, value):
        if not value.strip():
            raise ValueError('front_img cannot be empty string')
        return value

    @validator('profile_img')
    def validate_person_weight(cls, value):
        if not value.strip():
            raise ValueError('person_weight cannot be empty string')
        return value

def decode_base64_image(base64_string: str) -> np.ndarray:
    image_data = base64.b64decode(base64_string)
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = (
        cv2.imdecode(image_array, cv2.IMREAD_COLOR))
    return image

def encode_image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image)  # Encoding to JPEG
    image_bytes = buffer.tobytes()
    return base64.b64encode(image_bytes).decode('utf-8')

@app.post("/process-keypoints")
async def process_image_endpoint(request_data: ImageProcessingRequest):
    try:
        result = process_body_keypoints(request_data)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

def process_body_keypoints(request_data: ImageProcessingRequest):
    front_image = decode_base64_image(request_data.front_img)
    profile_image = decode_base64_image(request_data.profile_img)

    print("Front image: " + front_image[:100])
    print("Profile image: " + profile_image[:100])

    params = dict()
    params["model_folder"] = "/openpose/models/"
    params["net_resolution"] = "320x176"
    params["model_pose"] = "BODY_25"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    datum.cvInputData = front_image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    keypoints_front = datum.poseKeypoints.tolist() if datum.poseKeypoints is not None else None
    json.dump({"keypoints": keypoints_front}, indent=4)
    print("Body keypoints front: \n" + str(datum.poseKeypoints))
    skeleton_front = datum.cvOutputData

    datum.cvInputData = profile_image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    keypoints_profile = datum.poseKeypoints.tolist() if datum.poseKeypoints is not None else None
    json.dump({"keypoints": keypoints_profile}, indent=4)
    print("Body keypoints profile: \n" + str(datum.poseKeypoints))
    skeleton_profile = datum.cvOutputData

    skeleton_front_base64 = encode_image_to_base64(skeleton_front)
    skeleton_profile_base64 = encode_image_to_base64(skeleton_profile)

    return {"skeleton_front": skeleton_front_base64,
            "keypoints_front": keypoints_front,
            "skeleton_profile": skeleton_profile_base64,
            "keypoints_profile": keypoints_profile}
