from functools import cache
import os
import uuid
from focoos import Focoos, RuntimeTypes, DEV_API_URL, FocoosDetections
import gradio as gr
import numpy as np
from supervision import (
    LineZone,
    LineZoneAnnotator,
    Point,
    Position,
)
import supervision as sv
import cv2
import spaces

models = {
    "carrera1 (fcs_rtdetr)": "b4f61d27fc5c494f",
    "carrera2 (fcs_det_small)": "993aabb2bccb478d",
    "carrera3 (fcs_det_small)": "2d7eefc26d8b4e31",
    "carrera4 (fcs_det_small)": "11df512bdf8940bc",
    "carrera5 (fcs_det_small)": "2c92acb77e9e4ee9",
    "carrera6 (fcs_det_small)": "6ebb394a93134012",
}

SUBSAMPLE = 2
example_video = "./assets/test.mp4"
# TRACKER_WAIT_FRAMES = 200

tracker = sv.ByteTrack(
    frame_rate=30,
    minimum_matching_threshold=0.1,
    track_activation_threshold=0.1,
    lost_track_buffer=299,
)
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
line_annotator = LineZoneAnnotator(
    thickness=1,
    text_thickness=1,
    text_offset=5,
)

smoother = sv.DetectionsSmoother(length=1)


def focoos_to_sv(focoos_detections: FocoosDetections) -> sv.Detections:
    detections = focoos_detections.detections
    sv_detections = sv.Detections(
        xyxy=(
            np.array([det.bbox for det in detections])
            if len(detections) > 0
            else np.empty((0, 4))
        ),
        class_id=(
            np.array([det.cls_id for det in detections])
            if len(detections) > 0
            else np.array([])
        ),
        confidence=(
            np.array([det.conf for det in detections])
            if len(detections) > 0
            else np.array([])
        ),
        tracker_id=(
            np.array([det.cls_id for det in detections])
            if len(detections) > 0
            else np.array([])
        ),
    )
    return sv_detections


@cache
def load_model(model_name: str):
    client = Focoos(api_key=os.getenv("FOCOOS_API_KEY") or "", host_url=DEV_API_URL)
    model = client.get_local_model(
        model_ref=model_name,
        runtime_type=RuntimeTypes.ONNX_CUDA32,
    )
    return model


@spaces.GPU
def predict(
    video_path: str,
    model_name: str,
    threshold: float,
    x1,
    y1,
    x2,
    y2,
    progress=gr.Progress(),
):
    assert video_path is not None
    assert model_name is not None
    assert x1 is not None
    assert y1 is not None
    assert x2 is not None
    assert y2 is not None

    # line_counter = LineZone(
    #    start=Point(x=x1, y=y1),
    #    end=Point(x=x2, y=y2),
    #    triggering_anchors=[Position.BOTTOM_LEFT],
    # )
    progress(0, desc="load model and warmup...")
    model = load_model(models[model_name])
    progress(0.1, desc="predicting...")
    line_counter = LineZone(
        start=Point(x=x2, y=y2),
        end=Point(x=x1, y=y1),
        triggering_anchors=[Position.BOTTOM_LEFT, Position.BOTTOM_RIGHT],
    )
    classes = model.metadata.classes
    assert classes is not None
    cap = cv2.VideoCapture(video_path)

    # This means we will output mp4 videos
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    desired_fps = fps // SUBSAMPLE
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width > height:
        desired_width = 640
        desired_height = int((height / width) * 640)
    else:
        desired_height = 640
        desired_width = int((width / height) * 640)
    print(
        f"video: {video_path} fps: {fps}, desired_fps: {desired_fps}, width: {desired_width}, height: {desired_height}"
    )
    iterating, frame = cap.read()

    n_frames = 0

    # Use UUID to create a unique video file
    output_video_name = f"./output/output_{uuid.uuid4()}.mp4"
    print(output_video_name)

    # Output Video
    output_video = cv2.VideoWriter(output_video_name, video_codec, desired_fps, (desired_width, desired_height))  # type: ignore
    batch = []

    while iterating:
        if not cap.isOpened():
            print("Video ended")
            break

        if frame is None:
            continue
        frame = cv2.resize(frame, (desired_width, desired_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if n_frames % SUBSAMPLE == 0:
            batch.append(frame)
        if len(batch) == 2 * desired_fps:
            for frame in batch:
                res, _ = model.infer(frame, threshold=threshold, annotate=False)

                detections = focoos_to_sv(res)
                # detections = smoother.update_with_detections(detections)

                line_counter.trigger(detections=detections)
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
                line_annotator.annotate(
                    frame=annotated_frame, line_counter=line_counter
                )
                # frame = frame[:, :, ::-1].copy()
                output_video.write(annotated_frame[:, :, ::-1])
            batch = []
            output_video.release()

            yield output_video_name, {"latency(ms)": res.latency.get("inference")}
            output_video_name = f"./output/output_{uuid.uuid4()}.mp4"
            output_video = cv2.VideoWriter(output_video_name, video_codec, desired_fps, (desired_width, desired_height))  # type: ignore
        iterating, frame = cap.read()
        n_frames += 1


video_interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Video(),
        gr.Dropdown(list(models.keys())),
        gr.Slider(0, 1, 0.5),
        gr.Number(label="x1"),
        gr.Number(label="y1"),
        gr.Number(label="x2"),
        gr.Number(label="y2"),
    ],
    flagging_mode="manual",
    outputs=[gr.Video(streaming=True, autoplay=True, format="mp4"), gr.JSON()],
    examples=[
        [example_video, "carrera6 (fcs_det_small)", 0.6, 190, 280, 255, 340],
    ],
    description="Upload a video to track slot cars.",
)

live_rtsp_interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="RTSP URL"),
        gr.Dropdown(list(models.keys())),
        gr.Slider(0, 1, 0.5),
        gr.Number(label="x1"),
        gr.Number(label="y1"),
        gr.Number(label="x2"),
        gr.Number(label="y2"),
    ],
    outputs=[gr.Video(streaming=True, autoplay=True), gr.JSON()],
    description="Track slot cars from an RTSP stream.",
)


demo = gr.TabbedInterface(
    title="Slot car tracker (powered by FocoosAI)",
    interface_list=[video_interface, live_rtsp_interface],
    tab_names=["Video", "Live"],
)

if __name__ == "__main__":
    demo.launch(share=True)
