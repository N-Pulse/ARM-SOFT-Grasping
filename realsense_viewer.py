import numpy as np
import cv2
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 30
WINDOW = "RealSense D405  |  depth (left)  +  color (right)  |  q to quit"

def make_pipeline():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    print(f"Depth scale: {depth_sensor.get_depth_scale():.4f} m/unit")
    return pipeline

def run():
    pipeline = make_pipeline()
    align = rs.align(rs.stream.color)
    decimation = rs.decimation_filter()
    spatial    = rs.spatial_filter()
    temporal   = rs.temporal_filter()
    hole_fill  = rs.hole_filling_filter()
    colorizer  = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 9)
    snapshot_idx = 0

    try:
        for _ in range(30):
            pipeline.wait_for_frames()
        print("Streaming — press  q / Esc  to quit,  s  to snapshot.")
        while True:
            frames   = pipeline.wait_for_frames()
            aligned  = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_frame = decimation.process(depth_frame)
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_fill.process(depth_frame)
            depth_colored = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            color_image   = np.asanyarray(color_frame.get_data())
            depth_colored = cv2.resize(depth_colored, (color_image.shape[1], color_image.shape[0]))
            cx, cy = WIDTH // 2, HEIGHT // 2
            dist  = aligned.get_depth_frame().get_distance(cx, cy)
            label = f"centre: {dist*100:.1f} cm" if dist > 0 else "centre: --"
            for img in (depth_colored, color_image):
                cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.circle(img, (cx, cy), 6, (0, 255, 0), 2)
            combined = np.hstack((depth_colored, color_image))
            cv2.imshow(WINDOW, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                fname = f"snapshot_{snapshot_idx:04d}.png"
                cv2.imwrite(fname, combined)
                print(f"Saved {fname}")
                snapshot_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
