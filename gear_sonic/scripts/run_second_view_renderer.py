"""Publish a robot-only second_view from SONIC C++ state using MuJoCo Warp."""

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import msgpack
import tyro
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

GEAR_SONIC_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RendererConfig:
    """Addresses, render resources and target publication frequency."""

    state_host: str = "localhost"
    state_port: int = 5557
    publish_host: str = "0.0.0.0"
    publish_port: int = 5560
    fps: float = 50.0
    device: str = "cuda:0"
    model_path: Path = GEAR_SONIC_ROOT / "data/robot_model/model_data/g1/scene_43dof.xml"
    camera_config_path: Path = GEAR_SONIC_ROOT / "config/second_view_camera.json"


def main(config: RendererConfig):
    from gear_sonic.utils.inference.second_view_renderer import SecondViewRenderer

    renderer = SecondViewRenderer(config.model_path, config.camera_config_path, config.device)
    subscriber = ZMQStateSubscriber(host=config.state_host, port=config.state_port)
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 1)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(f"tcp://{config.publish_host}:{config.publish_port}")
    print(f"second_view PUB tcp://{config.publish_host}:{config.publish_port}; waiting for g1_debug", flush=True)
    period = 1.0 / config.fps
    report_start = None
    frames = 0
    render_s = encode_s = 0.0
    try:
        while True:
            started = time.monotonic()
            state = subscriber.get_msg()
            if state is None:
                time.sleep(period)
                continue
            if report_start is None:
                report_start = started
            received_at = time.time()
            rgb = renderer.render(state)
            rendered = time.monotonic()
            success, png = cv2.imencode(".png", rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
            assert success
            message = ImageMessageSchema(
                timestamps={"second_view": received_at},
                images={"second_view": png.tobytes()},
                metadata={"source_state_index": state["index"], "rendered_at": time.time()},
            ).serialize()
            publisher.send(msgpack.packb(message, use_bin_type=True))
            encoded = time.monotonic()
            frames += 1
            render_s += rendered - started
            encode_s += encoded - rendered
            elapsed = encoded - report_start
            if elapsed >= 5.0:
                print(
                    f"second_view: {frames / elapsed:.2f} FPS; "
                    f"render {1000 * render_s / frames:.2f} ms; "
                    f"PNG/publish {1000 * encode_s / frames:.2f} ms; "
                    f"state index {state['index']}", flush=True,
                )
                report_start = encoded
                frames = 0
                render_s = encode_s = 0.0
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("second_view renderer stopped", flush=True)
    finally:
        subscriber.close()
        publisher.close()
        context.term()
        renderer.close()


if __name__ == "__main__":
    main(tyro.cli(RendererConfig))
