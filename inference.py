"""Two-USB-camera MolmoAct2 inference. Dry-run is the safe default."""
import argparse, os, time
import cv2, numpy as np
from molmoact_so101.setup.robot import FollowerArm
from molmoact_so101.setup.usb_camera import CameraConfig, UsbCamera, camera_source, FLIP_CHOICES
from molmoact_so101.setup.frame_transforms import parse_joint_limits, parse_joint_offsets, parse_joint_signs
from molmoact_so101.model.runtime import AsyncPolicyRunner, RuntimeConfig
from molmoact_so101.model.backends import LocalPolicyBackend, RemotePolicyBackend, token_from_env

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True); p.add_argument("--head-cam", required=True); p.add_argument("--side-cam", required=True)
    p.add_argument("--width",type=int,default=640); p.add_argument("--height",type=int,default=480); p.add_argument("--fps",type=float,default=15)
    p.add_argument("--head-rotation",type=int,default=0); p.add_argument("--side-rotation",type=int,default=0); p.add_argument("--head-flip",choices=FLIP_CHOICES,default="none"); p.add_argument("--side-flip",choices=FLIP_CHOICES,default="none")
    p.add_argument("--camera-slots",default="head,side",help="model image order; only head,side or side,head")
    p.add_argument("--policy-backend",choices=("local","remote"),default="local"); p.add_argument("--server-url"); p.add_argument("--api-token-env",default="MOLMOACT_API_TOKEN"); p.add_argument("--request-timeout",type=float,default=10)
    mode=p.add_mutually_exclusive_group(); mode.add_argument("--dry-run",action="store_true"); mode.add_argument("--live",action="store_true")
    p.add_argument("--follower-port",default="/dev/ttyACM0"); p.add_argument("--device",default="cuda"); p.add_argument("--dtype",default="bfloat16"); p.add_argument("--num-steps",type=int,default=10); p.add_argument("--exec-hz",type=float,default=30); p.add_argument("--max-action-step-deg",type=float,default=15); p.add_argument("--max-observation-age-ms",type=float,default=500); p.add_argument("--action-watchdog-s",type=float,default=1); p.add_argument("--smooth-alpha",type=float,default=1.0,help="EMA target smoothing; 1.0 disables it. Test 0.7-0.85 only after latency is healthy."); p.add_argument("--show",action="store_true"); p.add_argument("--save-frames-dir"); p.add_argument("--joint-offsets",default="0,90,90,0,0,0"); p.add_argument("--joint-signs",default="1,-1,1,1,1,1"); p.add_argument("--joint-min"); p.add_argument("--joint-max")
    graph=p.add_mutually_exclusive_group(); graph.add_argument("--cuda-graph",dest="cuda_graph",action="store_true",help="Use CUDA graphs after server warm-up (default)."); graph.add_argument("--no-cuda-graph",dest="cuda_graph",action="store_false",help="Disable CUDA graphs for diagnosis."); p.set_defaults(cuda_graph=True)
    a=p.parse_args(); a.dry_run=not a.live; a.camera_slots=tuple(x.strip() for x in a.camera_slots.split(","))
    if set(a.camera_slots)!={"head","side"}: p.error("--camera-slots must contain head and side exactly once")
    if a.policy_backend=="remote" and not a.server_url: p.error("--server-url is required for remote policy backend")
    if not 0 < a.smooth_alpha <= 1: p.error("--smooth-alpha must be in (0, 1]")
    return a

def main():
    a=args(); head=side=follower=None
    try:
        head=UsbCamera(CameraConfig(camera_source(a.head_cam),a.width,a.height,a.fps,a.head_rotation,a.head_flip),"head")
        side=UsbCamera(CameraConfig(camera_source(a.side_cam),a.width,a.height,a.fps,a.side_rotation,a.side_flip),"side")
        if a.policy_backend=="local":
            from molmoact_so101.model.policy import MolmoActPolicy, REPO_ID
            backend=LocalPolicyBackend(MolmoActPolicy.from_pretrained(REPO_ID,device=a.device,dtype=a.dtype,apply_patches=a.dtype != "float32"))
        else: backend=RemotePolicyBackend(a.server_url,token_from_env(a.api_token_env),a.request_timeout)
        follower=FollowerArm(port=a.follower_port, simulate=a.dry_run)
        if a.live and follower.simulate: raise RuntimeError("live mode requires a connected SO-101; simulation is forbidden")
        if a.save_frames_dir: os.makedirs(a.save_frames_dir,exist_ok=True)
        cfg=RuntimeConfig(a.prompt,exec_hz=a.exec_hz,max_step_deg=a.max_action_step_deg,num_steps=a.num_steps,cuda_graph=a.cuda_graph,smooth_alpha=a.smooth_alpha,dry_run=a.dry_run,max_observation_age_ms=a.max_observation_age_ms,watchdog_s=a.action_watchdog_s,camera_slots=a.camera_slots,save_frames_dir=a.save_frames_dir)
        with AsyncPolicyRunner(backend=backend,follower=follower,head=head,side=side,signs=parse_joint_signs(a.joint_signs),offsets=parse_joint_offsets(a.joint_offsets),joint_min=parse_joint_limits(a.joint_min,-np.inf),joint_max=parse_joint_limits(a.joint_max,np.inf),config=cfg) as runner:
            while True:
                if a.show:
                    h,s=runner.latest_frames();
                    if h is not None: cv2.imshow("head",h)
                    if s is not None: cv2.imshow("side",s)
                    if cv2.waitKey(1)&255==ord("q"): break
                else: time.sleep(.1)
    except KeyboardInterrupt: pass
    finally:
        if follower: follower.request_torque(False); follower.disconnect()
        if head: head.close()
        if side: side.close()
        cv2.destroyAllWindows()
if __name__=="__main__": main()
