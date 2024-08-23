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
model = YOLOv10.from_pretrained("jameslahm/yolov10l").to(device)

filename = "video_demo.mp4"
result_file_path = f"./result_video.mp4"

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

        end=100,
    )
    all_detections = []

    with sv.VideoSink(result_file_path, video_info=video_info) as sink:
        for idx, frame in enumerate(frame_generator):
            img = Image.fromarray(frame)

            if idx == 16:
                # save to jpg
                print(frame)
                image_rgb = frame[:, :, ::-1]
                image_rgb.save("frame.jpg")
                break

            results = model(img)

            rr.set_time_sequence("frame", idx)

            rr.log("image", rr.Image(img).compress(jpeg_quality=85))

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

            logging.info(f"frame {idx} processed, detected {len(detections)} objects")

    return result_file_path, all_detections


print("---")
print("Device:", device)
print("Filename:", filename)
print("Result file path:", result_file_path)
print("---")

t0 = time.time()

setup_logging()

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

rr.save('recording.rrd')
print("Saved recording.rrd")
