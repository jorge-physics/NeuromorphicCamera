import cv2
import numpy as np
import time

def main():
    decay_tau = 0.050    # how long motion glows, ~50 ms
    gain = 4.0           # visual amplification
    motion_thresh = 0.05 # min per-frame brightness change to count as motion
    print("Press 'q' to quit. Press 's' to toggle label text set. Press 'l' to hide/show labels.")

    # -------------------------
    # Camera init
    # -------------------------
    cam_index = 1  # change to 1 (or 2, ...) if you want your external webcam
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {cam_index}.")
        return

    # OPTIONAL: ask for higher resolution / FPS if supported
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_FPS, 60)

    # Prime first frame
    ok, frame = cap.read()
    if not ok:
        print("Error: couldn't read first frame.")
        return

    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    H, W = gray_prev.shape

    # Trails for "event" viz
    ev_on  = np.zeros((H, W), dtype=np.float32)   # brightening events buffer
    ev_off = np.zeros((H, W), dtype=np.float32)   # darkening events buffer

    t_prev = time.time()

    # label_mode 0 -> STANDARD CAMERA / EVENT CAMERA
    # label_mode 1 -> FRAME-BASED / EVENT-BASED
    label_mode  = 0

    # show/hide labels
    show_labels = True

    while True:
        t_now = time.time()
        dt_loop = t_now - t_prev
        if dt_loop <= 0:
            dt_loop = 1e-6
        t_prev = t_now
        
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #cap.set(cv2.CAP_PROP_FPS, 30)

        ok, frame = cap.read()
        if not ok:
            print("Error: failed to grab frame.")
            break

        # grayscale normalized [0,1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # frame-to-frame diff
        diff = gray - gray_prev
        gray_prev = gray  # prevents long-term drift accumulation

        # separate positive/negative changes
        on_now  = np.clip(diff, 0, None)      # >0 brightening
        off_now = np.clip(-diff, 0, None)     # <0 darkening

        # suppress tiny single-frame jitters
        on_now[ on_now  < motion_thresh] = 0.0
        off_now[off_now < motion_thresh] = 0.0

        # exponential decay on the "trail" buffers
        decay_factor = np.exp(-dt_loop / decay_tau).astype(np.float32)
        ev_on  *= decay_factor
        ev_off *= decay_factor

        # inject this frame's activity
        ev_on  = np.maximum(ev_on,  on_now  * gain)
        ev_off = np.maximum(ev_off, off_now * gain)

        # clamp buffers
        ev_on  = np.clip(ev_on,  0.0, 1.0)
        ev_off = np.clip(ev_off, 0.0, 1.0)

        # build BGR visualization for events
        ev_viz = np.zeros((H, W, 3), dtype=np.float32)
        ev_viz[..., 2] = ev_on    # red channel (index 2)
        ev_viz[..., 0] = ev_off   # blue channel (index 0)
        ev_viz_uint8 = (ev_viz * 255).astype(np.uint8)

        # -------- PORTRAIT-FRIENDLY DISPLAY (STACKED) --------
        target_w = 640  # portrait width of each panel
        aspect   = frame.shape[0] / frame.shape[1]  # H / W from camera
        panel_h  = int(target_w * aspect)

        # resize camera view and event view
        frame_disp = cv2.resize(frame, (target_w, panel_h))
        ev_disp    = cv2.resize(ev_viz_uint8, (target_w, panel_h),
                                 interpolation=cv2.INTER_NEAREST)

        # Pick which label strings we want (2 modes)
        if label_mode == 0:
            top_label    = "STANDARD CAMERA"
            bottom_label = "NEUROMORPHIC CAMERA"
            top_color    = (0,255,0)   # green-ish
            bot_color    = (0,0,255)   # red-ish/magenta-ish text in BGR
        else:
            top_label    = "FRAME-BASED"
            bottom_label = "EVENT-BASED"
            top_color    = (0,255,0)
            bot_color    = (0,0,255)

        # Only draw labels if show_labels is True
        if show_labels:
            # cv2.putText(frame_disp, top_label, (10,30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #             top_color, 2, cv2.LINE_AA)
            # cv2.putText(ev_disp, bottom_label, (10,30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #             bot_color, 2, cv2.LINE_AA)
            draw_centered_label(frame_disp, top_label, top_color)
            draw_centered_label(ev_disp, bottom_label, bot_color)


        # vertical stack -> portrait frame
        portrait_frame = np.vstack([frame_disp, ev_disp])

        # show it
        cv2.imshow("Portrait Preview", portrait_frame)

        # read keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # toggle label set
            label_mode = 1 - label_mode
        elif key == ord('l'):
            # toggle visibility
            show_labels = not show_labels

    cap.release()
    cv2.destroyAllWindows()
    
    # --- draw centered label on a panel ---
def draw_centered_label(img, text, color):
    # img: the panel you're annotating (e.g. frame_disp or ev_disp)
    # text: string
    # color: (B,G,R)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale= 0.8
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # center horizontally
    x = (img.shape[1] - text_w) // 2

    # a little down from the top; 10 px padding looks decent
    y = 10 + text_h

    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    main()
