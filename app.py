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

SUBSAMPLE = 2
example_video1 = "./assets/test.mp4"
example_video2 = "./assets/test2.mp4"
# TRACKER_WAIT_FRAMES = 200

tracker = sv.ByteTrack(
    frame_rate=30,
    minimum_matching_threshold=0.1,
    track_activation_threshold=0.1,
    lost_track_buffer=299,
)
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(
    border_radius=10,
)
line_annotator = LineZoneAnnotator(
    thickness=1,
    text_thickness=1,
    text_offset=5,
    display_in_count=False,
    display_out_count=False,
)

triangle_annotator = sv.TriangleAnnotator()
corner_annotator = sv.BoxCornerAnnotator(thickness=2, corner_length=6)


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
            car = self.cars.setdefault(
                car_id, CarStats(name=self.classes_labels[car_id], id=car_id)
            )
            if count != car.total_laps:  # new lap
                car.total_laps = count
                if car.initial_lap_frame is not None:
                    time = round((frame_n - car.initial_lap_frame) / self.fps, 2)
                    print(
                        f"new lap time: {time}s initial lap frame: {car.initial_lap_frame} frame: {frame_n} elapsed frames: {frame_n - car.initial_lap_frame}"
                    )
                    car.last_lap_time = time
                    if car.best_lap_time is None or time < car.best_lap_time:
                        car.best_lap_time = time
                car.initial_lap_frame = frame_n

    def get_total_laps(self) -> int:
        return max(car.total_laps for car in self.cars.values()) if self.cars else 0

    def get_best_lap_time_and_car(self) -> tuple[Optional[float], Optional[str]]:
        _best_lap_time = None
        _best_car_name = None
        for car in self.cars.values():
            if car.best_lap_time is not None and (
                _best_lap_time is None or car.best_lap_time < _best_lap_time
            ):
                _best_lap_time = car.best_lap_time
                _best_car_name = car.name
        return _best_lap_time, _best_car_name if _best_lap_time is not None else None

    def _put_text_with_border(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple,
        font_scale: float,
        thickness: int,
        border_color: tuple[int, int, int] = (0, 0, 0),
        border_thickness: int = 1,
        text_color: tuple[int, int, int] = (255, 255, 255),
        text_thickness: int = 1,
    ):
        text_size, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
        )
        text_x, text_y = position
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            border_color,
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        return text_size

    def annotate(self, frame: np.ndarray, frame_n: int) -> np.ndarray:
        current_time_text = f"Race time: {round(float(frame_n / self.fps), 1)}s"
        total_laps_text = f"Total laps: {self.get_total_laps()}"

        text_y = 30
        text_size_time = self._put_text_with_border(
            frame, current_time_text, (10, text_y), 1, 1
        )
        text_y += text_size_time[1] + 10
        text_size_laps = self._put_text_with_border(
            frame, total_laps_text, (10, text_y), 1, 1
        )
        text_y += text_size_laps[1] + 10
        best_lap_time, best_car_name = self.get_best_lap_time_and_car()
        best_lap_time_text = f"best lap: {best_lap_time if best_lap_time is not None else 'N/A'}s {best_car_name if best_car_name is not None else ''}"
        text_size_best_lap = self._put_text_with_border(
            frame,
            best_lap_time_text,
            (10, text_y),
            1,
            1,
            text_color=(128, 0, 128),
            border_color=(128, 0, 128),
            border_thickness=2,
        )
        text_y += text_size_best_lap[1] + 10

        for car in self.cars.values():
            text_name = f"{car.name}"
            current_lap_time = f"current lap: {round(float((frame_n - car.initial_lap_frame)) / self.fps, 1)}s"
            text_laps = f"total laps: {car.total_laps}"
            text_last_lap = f"last lap: {car.last_lap_time if car.last_lap_time is not None else 'N/A'}s"
            text_best_lap = f"best lap: {car.best_lap_time if car.best_lap_time is not None else 'N/A'}s"

            text_size_name = self._put_text_with_border(
                frame, text_name, (10, text_y), 0.5, 1
            )
            text_y += text_size_name[1] + 10
            self._put_text_with_border(frame, current_lap_time, (20, text_y), 0.5, 1)
            text_y += text_size_name[1] + 5
            self._put_text_with_border(frame, text_laps, (20, text_y), 0.5, 1)
            text_y += text_size_name[1] + 5
            self._put_text_with_border(frame, text_last_lap, (20, text_y), 0.5, 1)
            text_y += text_size_name[1] + 5
            self._put_text_with_border(frame, text_best_lap, (20, text_y), 0.5, 1)
            text_y += text_size_name[1] + 5

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
    assert video_path is not None, "video_path is required"
    assert model_name is not None, "model_name is required"
    assert x1 is not None, "x1 is required"
    assert y1 is not None, "y1 is required"
    assert x2 is not None, "x2 is required"
    assert y2 is not None, "y2 is required"

    progress(0, desc="load model and warmup...")
    model = load_model(models[model_name])
    progress(0.1, desc="predicting first batch...")
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # if width > height:
    #     desired_width = 640
    #     desired_height = int((height / width) * 640)
    # else:
    #     desired_height = 640
    #     desired_width = int((width / height) * 640)

    desired_height = height
    desired_width = width
    print(
        f"video: {video_path} fps: {fps}, total_frames: {total_frames}, desired_fps: {desired_fps}, width: {desired_width}, height: {desired_height}"
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
        res, _ = model.infer(frame, threshold=threshold, annotate=False)

        detections = focoos_to_sv(res)

        line_counter.trigger(detections=detections)
        track_stats.update(line_counter, n_frames)
        labels = [
            f"{classes_labels[int(class_id)]}"
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
            best_lap_time, best_car_name = track_stats.get_best_lap_time_and_car()
            yield output_video_name, {
                "latency(ms)": res.latency.get("inference"),
                "total_laps": track_stats.get_total_laps(),
                "best_lap_time": f"{best_lap_time}s {best_car_name}",
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
    flagging_mode="never",
    outputs=[gr.Video(streaming=True, autoplay=True, format="mp4"), gr.JSON()],
    examples=[
        [
            example_video1,
            "carrera6 (fcs_det_small)",
            0.6,
            380,
            560,
            510,
            680,
        ],  # 380, 560, 510, 680
        [example_video2, "carrera6 (fcs_det_small)", 0.6, 515, 620, 600, 540],
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
    allow_flagging="never",
)


demo = gr.TabbedInterface(
    title="Slot car tracker (powered by FocoosAI)",
    interface_list=[video_interface, live_rtsp_interface],
    tab_names=["Video", "Live"],
)

if __name__ == "__main__":
    demo.launch(share=True)
