import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
@torch.no_grad()
def detect(weights='yolov5s.pt',  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # ---------------------- COUNT_CROPPED -------------------------------

    COUNT_CROPPED = 0

    # --------------------------------------------------------------------

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # Process detections

        for i, det in enumerate(pred):  # detections per image
            # ---------------------- Check valid cars flag -----------------------

            have_valid_cars = False

            # --------------------------------------------------------------------
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # ---------------------- Check allows classes -----------------------

                machine_xyxy   = []
                machine_conf   = []
                machine_class  = []
                
                #check and save valid cars
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls) 
                    float_conf = float(conf)

                    if names[c] in ['car', 'truck', 'bus']:
                        if (float_conf > 0.7):
                            width_bbox = abs(int(xyxy[2]) - int(xyxy[0]))
                            height_bbox = abs(int(xyxy[3]) - int(xyxy[1]))

                            if width_bbox > 200 and height_bbox > 200:
                                machine_xyxy.append(xyxy)
                                machine_conf.append(conf)
                                machine_class.append(cls)
                                have_valid_cars = True
                
                if len(machine_xyxy) == 0:
                    continue

                not_overlap_machine_xyxy   = []
                not_overlap_machine_conf   = []
                not_overlap_machine_class  = []

                for i in range(0, len(machine_xyxy)):
                    overlap = False

                    #bbox machine cords
                    machine_min_width  = min(int(machine_xyxy[i][0]), int(machine_xyxy[i][2]))
                    machine_max_width  = max(int(machine_xyxy[i][0]), int(machine_xyxy[i][2]))
                    machine_min_height = min(int(machine_xyxy[i][1]), int(machine_xyxy[i][3]))
                    machine_max_height = max(int(machine_xyxy[i][1]), int(machine_xyxy[i][3]))

                    # print("machine_min_width: ", machine_min_width)
                    # print("machine_max_width: ", machine_max_width)
                    # print("machine_min_height: ", machine_min_height)
                    # print("machine_max_height: ", machine_max_height)

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        # if not names[c] in ['car', 'truck', 'bus']:
                        if not (c == int(machine_class[i]) and xyxy == machine_xyxy[i]):
                            #bbox other object cords
                            object_min_width  = min(int(xyxy[0]), int(xyxy[2]))
                            object_max_width  = max(int(xyxy[0]), int(xyxy[2]))
                            object_min_height = min(int(xyxy[1]), int(xyxy[3]))
                            object_max_height = max(int(xyxy[1]), int(xyxy[3]))
                            
                            # print("object_min_width: ", object_min_width)
                            # print("object_max_width: ", object_max_width)
                            # print("object_min_height: ", object_min_height)
                            # print("object_max_height: ", object_max_height)

                            if  machine_max_width >= object_min_width and machine_min_width <= object_max_width and machine_min_height <= object_max_height and machine_max_height >= object_min_height:
                                overlap = True
                                break

                    if overlap == False:
                        not_overlap_machine_xyxy.append(machine_xyxy[i])
                        not_overlap_machine_conf.append(machine_conf[i])
                        not_overlap_machine_class.append(machine_class[i])

                if len(not_overlap_machine_xyxy) == 0:
                    continue

                if have_valid_cars:
                    for i in range(0, len(not_overlap_machine_xyxy)):

                        if save_img or save_crop or view_img:  # Add bbox to image
                            COUNT_CROPPED += 1
                            print("Cropped: ", COUNT_CROPPED + 1, 'imgs')

                            label = None if hide_labels else (names[int(not_overlap_machine_class[i])] if hide_conf else f'{names[int(not_overlap_machine_class[i])]} {not_overlap_machine_conf[i]:.2f}')

                            if save_img:
                                plot_one_box(not_overlap_machine_xyxy[i], im0, label=label, color=colors(int(not_overlap_machine_class[i]), True), line_thickness=line_thickness)
                                # print("plot_one_box: ", plot_one_box)

                            if save_crop:                                        
                                save_one_box(not_overlap_machine_xyxy[i], imc, file=save_dir / 'crops' / names[int(not_overlap_machine_class[i])] / f'{p.stem}.jpg', BGR=True, gain=1.15)

            if have_valid_cars:
                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
                    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    print(f'Done. ({time.time() - t0:.3f}s)')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'thop'))
    detect(**vars(opt))
