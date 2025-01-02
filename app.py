from functools import cache
import os
from pathlib import Path
import uuid
from focoos import Focoos, RuntimeTypes, DEV_API_URL, FocoosDetections
import gradio as gr
import numpy as np
from supervision import VideoInfo, VideoSink, get_video_frames_generator
import supervision as sv
from PIL import Image
import cv2

MODEL = "2d7eefc26d8b4e31"
SUBSAMPLE = 2
example_video = "./assets/test.mp4"
runtime_types = [
    RuntimeTypes.ONNX_CUDA32,
    RuntimeTypes.ONNX_TRT32,
    RuntimeTypes.ONNX_TRT16,
    RuntimeTypes.ONNX_CPU,
]

tracker = sv.ByteTrack(
    frame_rate=100,
    # minimum_matching_threshold=0.3,
    # track_activation_threshold=0.05,
    # lost_track_buffer=60,
)
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


def focoos_to_sv(focoos_detections: FocoosDetections) -> sv.Detections:
    detections = focoos_detections.detections
    sv_detections = sv.Detections(
        xyxy=np.array([det.bbox for det in detections]),
        class_id=np.array([det.cls_id for det in detections]),
        confidence=np.array([det.conf for det in detections]),
        tracker_id=np.array([det.cls_id for det in detections]),
    )
    return sv_detections


@cache
def load_model(runtime_type: RuntimeTypes):
    client = Focoos(api_key=os.getenv("FOCOOS_API_KEY") or "", host_url=DEV_API_URL)
    model = client.get_local_model(
        model_ref=MODEL,
        runtime_type=runtime_type,
    )
    return model


def predict(
    video_path: str,
    runtime_type: RuntimeTypes,
    threshold: float,
    progress=gr.Progress(),
):

    progress(0, desc="load model and warmup...")
    model = load_model(runtime_type)
    progress(0.1, desc="predicting...")

    classes = model.metadata.classes
    assert classes is not None
    cap = cv2.VideoCapture(video_path)

    # This means we will output mp4 videos
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    desired_fps = fps // SUBSAMPLE
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

    iterating, frame = cap.read()

    n_frames = 0

    # Use UUID to create a unique video file
    output_video_name = f"./output/output_{uuid.uuid4()}.mp4"
    print(output_video_name)

    # Output Video
    output_video = cv2.VideoWriter(output_video_name, video_codec, desired_fps, (width, height))  # type: ignore
    batch = []

    while iterating:
        iterating, frame = cap.read()
        n_frames += 1
        if frame is None:
            continue
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if n_frames % SUBSAMPLE == 0:
            batch.append(frame)
        if len(batch) == 2 * desired_fps:
            for frame in batch:
                res, _ = model.infer(frame, threshold=threshold, annotate=False)

                detections = focoos_to_sv(res)

                labels = [
                    f"{classes[int(class_id)]}: {confid*100:.0f}%"
                    for class_id, confid in zip(detections.class_id, detections.confidence)  # type: ignore
                ]

                annotated_frame = bounding_box_annotator.annotate(
                    scene=frame.copy(), detections=detections
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )
                # frame = frame[:, :, ::-1].copy()
                output_video.write(annotated_frame[:, :, ::-1])
            batch = []
            output_video.release()
            yield output_video_name, {"latency": res.latency.get("inference")}
            output_video_name = f"./output/output_{uuid.uuid4()}.mp4"
            output_video = cv2.VideoWriter(output_video_name, video_codec, desired_fps, (width, height))  # type: ignore


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Video(), gr.Dropdown(runtime_types), gr.Slider(0, 1, 0.5)],
    outputs=[gr.Video(streaming=True, autoplay=True), gr.JSON()],
    examples=[
        [example_video, RuntimeTypes.ONNX_CUDA32, 0.5],
        [example_video, RuntimeTypes.ONNX_TRT32, 0.5],
        [example_video, RuntimeTypes.ONNX_TRT16, 0.5],
    ],
    title="Focoos Model Inference",
    description="Upload an image to get detections from the Focoos model.",
)

if __name__ == "__main__":
    demo.launch(share=True)
