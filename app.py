from dataclasses import dataclass
from functools import cache
import os
import time
from typing import Optional
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

SUBSAMPLE = 10
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
    display_in_count=False,
    display_out_count=False,
)

smoother = sv.DetectionsSmoother(length=1)


@dataclass
class CarStats:
    name: str
    id: int
    total_laps: int = 0
    initial_lap_frame: Optional[int] = None
    last_lap_time: Optional[float] = None
    best_lap_time: Optional[float] = None


class TrackStats:
    def __init__(self, classes_labels: list[str], fps: int):
        self.cars = {}
        self.classes_labels = classes_labels
        self.fps = fps

    def update(self, line_counter: LineZone, frame_n: int):
        for car_id, count in line_counter.in_count_per_class.items():
            if car_id not in self.cars:
                self.cars[car_id] = CarStats(
                    name=self.classes_labels[car_id], id=car_id
                )

            if count != self.cars[car_id].total_laps:  # new lap
                self.cars[car_id].total_laps = count

                if self.cars[car_id].initial_lap_frame is not None:
                    time = round(
                        float(
                            (frame_n - self.cars[car_id].initial_lap_frame) / self.fps
                        ),
                        5,
                    )
                    print(
                        f"new lap time: {time}s initial lap frame: {self.cars[car_id].initial_lap_frame} frame: {frame_n} elapsed frames: {frame_n - self.cars[car_id].initial_lap_frame}"
                    )
                    self.cars[car_id].last_lap_time = time
                self.cars[car_id].initial_lap_frame = frame_n

    def get_best_lap_time(self) -> tuple[Optional[float], Optional[str]]:
        best_lap_time = None
        best_car_name = None
        for car in self.cars.values():
            if car.last_lap_time is not None and (
                best_lap_time is None or car.last_lap_time < best_lap_time
            ):
                best_lap_time = round(car.last_lap_time, 2)
                best_car_name = car.name
        return best_lap_time, best_car_name

    def get_total_laps(self) -> int:
        return max(car.total_laps for car in self.cars.values()) if self.cars else 0

    def annotate(self, frame: np.ndarray, frame_n: int) -> np.ndarray:
        current_time_text = f"race time: {round(float(frame_n / self.fps), 1)}"
        text_size_time, _ = cv2.getTextSize(
            current_time_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1
        )
        text_y = text_size_time[1] + 10
        cv2.putText(
            frame,
            current_time_text,
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for car in self.cars.values():

            text_laps = f"{car.name}: {car.total_laps} laps"
            text_time = f"{car.name}: {str(car.last_lap_time)} s"
            text_size_laps, _ = cv2.getTextSize(
                text_laps, cv2.FONT_HERSHEY_SIMPLEX, 1, 1
            )
            text_size_time, _ = cv2.getTextSize(
                text_time, cv2.FONT_HERSHEY_SIMPLEX, 1, 1
            )
            text_x = frame.shape[1] - text_size_laps[0] - 10
            text_y = text_size_laps[1] + 10
            cv2.putText(
                frame,
                text_laps,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                text_time,
                (text_x, text_y + text_size_laps[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame


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
        triggering_anchors=[Position.BOTTOM_LEFT],
    )
    classes_labels = model.metadata.classes
    assert classes_labels is not None

    cap = cv2.VideoCapture(video_path)

    # This means we will output mp4 videos
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    desired_fps = fps
    track_stats = TrackStats(classes_labels, fps)

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
        # print(f"frame: {n_frames}")
        res, _ = model.infer(frame, threshold=threshold, annotate=False)

        detections = focoos_to_sv(res)
        # detections = smoother.update_with_detections(detections)

        line_counter.trigger(detections=detections)
        track_stats.update(line_counter, n_frames)
        labels = [
            f"{classes_labels[int(class_id)]}: {confid*100:.0f}%"
            for class_id, confid in zip(detections.class_id, detections.confidence)  # type: ignore
        ]

        annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)
        # Add label with total laps in the top right corner of the frame
        annotated_frame = track_stats.annotate(annotated_frame, n_frames)
        batch.append(annotated_frame.copy())

        if len(batch) == SUBSAMPLE * desired_fps:
            output_video = cv2.VideoWriter(output_video_name, video_codec, desired_fps, (desired_width, desired_height))  # type: ignore

            for frame in batch:
                output_video.write(frame[:, :, ::-1])

            output_video.release()
            print(f"writed video: {output_video_name} batch size: {len(batch)}")
            yield output_video_name, {
                "latency(ms)": res.latency.get("inference"),
                "total_laps": track_stats.get_total_laps(),
                "best_lap_time": track_stats.get_best_lap_time(),
            }
            batch = []
            output_video_name = f"./output/output_{uuid.uuid4()}.mp4"
        iterating, frame = cap.read()
        n_frames += 1
    cap.release()
    print("done")
    return


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
