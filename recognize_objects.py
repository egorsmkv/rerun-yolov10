"""
Code from this notebook:

https://colab.research.google.com/drive/1zWQB1amX0v8f9AJj9wiTdcWPx8GUKQj9?usp=sharing

Join our community about **Computer Vision**: https://t.me/computer_vision_uk and https://discord.gg/yVAjkBgmt4
"""

import time
import logging
from os import remove
from os.path import exists

import torch
import numpy as np
import supervision as sv
import rerun as rr

from ultralytics import YOLOv10
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# yolov10{n/s/m/b/l/x}
model = YOLOv10.from_pretrained("jameslahm/yolov10x").to(device)

model_names = {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush",
}

# config
check_confidence = True
min_confidence = 0.8

filename = "video_demo.mp4"
result_file_path = "result_video.mp4"

rr.init("rerun_yolov10")

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
TRACKER = sv.ByteTrack()


def setup_logging():
    logger = logging.getLogger()
    rerun_handler = rr.LoggingHandler("logs")
    rerun_handler.setLevel(-1)
    logger.addHandler(rerun_handler)


def annotate_image(input_image, detections, labels) -> np.ndarray:
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


def process_video(
    input_video,
):
    video_info = sv.VideoInfo.from_video_path(input_video)
    frame_generator = sv.get_video_frames_generator(
        source_path=input_video,
        # limit the number of frames to process
        # end=100,
    )
    all_detections = []
    all_tracker_ids = []

    with sv.VideoSink(result_file_path, video_info=video_info) as sink:
        for idx, frame in enumerate(frame_generator):
            img = Image.fromarray(frame)

            results = model(img)

            rr.set_time_sequence("frame", idx)
            rr.log("image", rr.Image(frame[:, :, ::-1]).compress(jpeg_quality=85))

            final_labels = []

            detections = sv.Detections.from_ultralytics(results[0])
            detections = TRACKER.update_with_detections(detections)

            for class_id_idx, _ in enumerate(detections.class_id.tolist()):
                label = detections.data["class_name"][class_id_idx]
                final_labels.append(label)

            annotated_frame = annotate_image(
                input_image=frame,
                detections=detections,
                labels=final_labels,
            )
            all_detections.append((idx, detections))
            sink.write_frame(annotated_frame)

            # rr.log("image", rr.Image(annotated_frame[:, :, ::-1]).compress(jpeg_quality=85))

            logging.info(f"frame {idx} processed, detected {len(detections)} objects")

            print(detections)

            # Extend all_tracker_ids
            for tracker_id in detections.tracker_id:
                if tracker_id not in all_tracker_ids:
                    all_tracker_ids.append(tracker_id)

            # Log all detections
            for idx, tracked_id in enumerate(detections.tracker_id):
                xyxy = detections.xyxy[idx]
                confidence = detections.confidence[idx]
                class_ids = detections.class_id[idx]

                if check_confidence and confidence < min_confidence:
                    continue

                rr.log(
                    f"image/tracked/{tracked_id}",
                    rr.Boxes2D(
                        array=xyxy,
                        array_format=rr.Box2DFormat.XYXY,
                        class_ids=class_ids,
                    ),
                )

            # Remove non visible trackers
            for tracker_id in all_tracker_ids:
                if tracker_id not in detections.tracker_id:
                    rr.log(f"image/tracked/{tracker_id}", rr.Clear(recursive=False))

    return result_file_path, all_detections


print("---")
print("Device:", device)
print("Filename:", filename)
print("Result file path:", result_file_path)
print("---")

t0 = time.time()

setup_logging()

class_descriptions = [rr.AnnotationInfo(id=k, label=v) for k, v in model_names.items()]
rr.log("/", rr.AnnotationContext(class_descriptions), static=True)

saved_filename, all_detections = process_video(filename)

print("---")
print("Result file:", saved_filename)
print("---")

print("Some detections:")
print(all_detections[:1])

print("Time elapsed:", time.time() - t0)

if exists("recording.rrd"):
    remove("recording.rrd")
    print("Removed recording.rrd")

rr.save("recording.rrd")
print("Saved recording.rrd")
