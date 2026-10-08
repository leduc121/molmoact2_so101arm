#!/usr/bin/env python3
"""Camera-only USB validation; it never imports robot or model modules."""
import argparse, os, sys, time
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from molmoact_so101.setup.usb_camera import CameraConfig, UsbCamera, camera_source, FLIP_CHOICES
def main():
 p=argparse.ArgumentParser(); p.add_argument("--list-cameras",action="store_true"); p.add_argument("--head-cam"); p.add_argument("--side-cam"); p.add_argument("--width",type=int,default=640); p.add_argument("--height",type=int,default=480); p.add_argument("--fps",type=float,default=15); p.add_argument("--head-rotation",type=int,default=0); p.add_argument("--side-rotation",type=int,default=0); p.add_argument("--head-flip",choices=FLIP_CHOICES,default="none"); p.add_argument("--side-flip",choices=FLIP_CHOICES,default="none"); p.add_argument("--save-dir"); p.add_argument("--no-show",action="store_true",help="capture without OpenCV GUI; useful for headless OpenCV"); p.add_argument("--frames",type=int,default=None,help="exit after this many paired captures")
 a=p.parse_args()
 if a.list_cameras:
  for root,dirs,files in os.walk("/dev/v4l/by-id") if os.path.isdir("/dev/v4l/by-id") else []: [print(os.path.join(root,f),"->",os.path.realpath(os.path.join(root,f))) for f in files]
  print("video devices:", [f"/dev/{x}" for x in os.listdir("/dev") if x.startswith("video")]); return
 if not a.head_cam or not a.side_cam: p.error("--head-cam and --side-cam are required unless --list-cameras")
 if a.save_dir: os.makedirs(a.save_dir,exist_ok=True)
 h=UsbCamera(CameraConfig(camera_source(a.head_cam),a.width,a.height,a.fps,a.head_rotation,a.head_flip),"head"); s=UsbCamera(CameraConfig(camera_source(a.side_cam),a.width,a.height,a.fps,a.side_rotation,a.side_flip),"side"); count=0; start=time.monotonic(); show=not a.no_show; saved=False; last_report=0.
 try:
  while True:
   x,y=h.read(),s.read(); count+=1
   if not x or not y: print("capture failure"); continue
   now=time.monotonic(); fps=count/(now-start)
   if now-last_report >= 1: print(f"head={x.bgr.shape[1]}x{x.bgr.shape[0]} side={y.bgr.shape[1]}x{y.bgr.shape[0]} capture={fps:.1f} fps"); last_report=now
   if a.save_dir and not saved: cv2.imwrite(os.path.join(a.save_dir,"head.jpg"),x.bgr); cv2.imwrite(os.path.join(a.save_dir,"side.jpg"),y.bgr); print(f"saved samples to {a.save_dir}"); saved=True
   if a.frames is not None and count >= a.frames: break
   if not show: 
    if saved: break
    continue
   try:
    cv2.putText(x.bgr,f"head {x.bgr.shape[1]}x{x.bgr.shape[0]} {fps:.1f} fps",(10,30),0,.6,(0,255,0),2); cv2.putText(y.bgr,"side",(10,30),0,.6,(0,255,0),2); cv2.imshow("head",x.bgr); cv2.imshow("side",y.bgr); key=cv2.waitKey(1)&255
   except cv2.error:
    print("OpenCV GUI unavailable; continuing headless. Use --save-dir to retain samples."); show=False; continue
   if key==ord("s") and a.save_dir: cv2.imwrite(os.path.join(a.save_dir,"head.jpg"),x.bgr); cv2.imwrite(os.path.join(a.save_dir,"side.jpg"),y.bgr)
   if key in (ord("q"),27): break
 finally:
  h.close(); s.close()
  try: cv2.destroyAllWindows()
  except cv2.error: pass
if __name__=="__main__": main()
