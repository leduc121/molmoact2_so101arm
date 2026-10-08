"""Fresh-observation async runner. It never holds or replays a stale action."""
from __future__ import annotations
import collections, os, threading, time, uuid
from dataclasses import dataclass
from typing import Optional
import cv2, numpy as np
from PIL import Image
from .protocol import PolicyRequest, encode_jpeg, validate_actions
from ..setup.frame_transforms import JOINT_COUNT, clip_action

# Training/action playback rate. Kept here so remote clients do not import model deps.
ACTION_FPS = 30.0

def _pil(bgr): return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

@dataclass
class RuntimeConfig:
    prompt: str; exec_hz: float = 30.; max_step_deg: float = 15.; actions_per_chunk: Optional[int] = None
    smooth_alpha: float = 1.; ensemble_m: float = .5; warmup_predictions: int = 0; num_steps: int = 10
    cuda_graph: bool = True; save_frames_dir: Optional[str] = None; dry_run: bool = True
    max_observation_age_ms: float = 500.; watchdog_s: float = 1.; camera_slots: tuple[str,str] = ("head", "side")

class ChunkRingBuffer:
    def __init__(self, capacity=6): self.lock=threading.Lock(); self.entries=collections.deque(maxlen=capacity); self.next_id=1
    def add(self, chunk, t_obs):
        with self.lock:
            ident=self.next_id; self.next_id+=1; self.entries.append((chunk,t_obs,ident)); return ident
    def snapshot(self):
        with self.lock: return list(self.entries)
    def clear(self):
        with self.lock: self.entries.clear()

class FrameBuffer:
    def __init__(self): self.lock=threading.Lock(); self.head=self.side=None
    def set(self, head, side):
        with self.lock: self.head,self.side=head,side
    def get(self):
        with self.lock: return self.head,self.side

class _Producer(threading.Thread):
    def __init__(self, backend, follower, head, side, ring, frames, signs, offsets, joint_min, joint_max, cfg):
        super().__init__(daemon=True, name="InferenceProducer"); self.backend=backend; self.follower=follower; self.head=head; self.side=side; self.ring=ring; self.frames=frames; self.signs=signs; self.offsets=offsets; self.joint_min=joint_min; self.joint_max=joint_max; self.cfg=cfg; self.stop_event=threading.Event(); self.failed=threading.Event()
    def stop(self): self.stop_event.set()
    def run(self):
        count=0
        while not self.stop_event.is_set():
            try:
                h,s=self.head.read(),self.side.read()
                if h is None or s is None: self.ring.clear(); self.failed.set(); time.sleep(.1); continue
                age=(time.monotonic()-min(h.captured_at,s.captured_at))*1000
                if age > self.cfg.max_observation_age_ms: self.ring.clear(); self.failed.set(); continue
                self.frames.set(h.bgr,s.bgr); arm=self.follower.get_state().astype(np.float32)
                if arm.shape != (JOINT_COUNT,) or not np.isfinite(arm).all(): raise ValueError("invalid robot joint state")
                ordered={"head":h.bgr,"side":s.bgr}; images=[ordered[x] for x in self.cfg.camera_slots]
                if self.cfg.save_frames_dir:
                    stamp=str(int(time.time()*1000)); [cv2.imwrite(os.path.join(self.cfg.save_frames_dir,f"{stamp}_in{i}.jpg"), im) for i,im in enumerate(images)]
                req=PolicyRequest(str(uuid.uuid4()), self.cfg.prompt, self.signs*arm+self.offsets, time.monotonic(), [encode_jpeg(_pil(im)) for im in images], self.cfg.camera_slots, self.cfg.num_steps, self.cfg.cuda_graph)
                raw=validate_actions(self.backend.predict(req))
                response_age_ms = (time.monotonic() - req.observation_monotonic) * 1000.0
                if response_age_ms > self.cfg.max_observation_age_ms:
                    raise RuntimeError(f"stale inference response ({response_age_ms:.0f}ms > {self.cfg.max_observation_age_ms:.0f}ms)")
                actions=np.clip((raw-self.offsets)*self.signs,self.joint_min,self.joint_max)
                count+=1; self.failed.clear()
                diag = getattr(self.backend, "last_diagnostics", {})
                print(f"[Producer] id={count} age={response_age_ms:.0f}ms transport={diag.get('transport_ms', 0):.0f}ms model={diag.get('policy_ms', 0) or 0:.0f}ms chunk={actions.shape[0]}")
                if count>self.cfg.warmup_predictions: self.ring.add(actions, req.observation_monotonic)
            except Exception as exc:
                self.ring.clear(); self.failed.set(); print(f"[Producer] {type(exc).__name__}: {exc}"); time.sleep(.1)

class _Consumer(threading.Thread):
    def __init__(self, follower, ring, producer, cfg):
        super().__init__(daemon=True, name="ExecutionConsumer"); self.follower=follower; self.ring=ring; self.producer=producer; self.cfg=cfg; self.stop_event=threading.Event(); self.last=None
    def stop(self): self.stop_event.set()
    def run(self):
        interval=1/self.cfg.exec_hz
        while not self.stop_event.is_set():
            now=time.monotonic()
            if self.producer.failed.is_set(): time.sleep(interval); continue
            active=[]; ages=[]
            for chunk,t_obs,_ in self.ring.snapshot():
                n=min(len(chunk),self.cfg.actions_per_chunk or len(chunk)); step=int((now-t_obs)*ACTION_FPS)
                if 0 <= step < n and now-t_obs <= self.cfg.watchdog_s: active.append(chunk[step]); ages.append(now-t_obs)
            if active:
                w=np.exp(-self.cfg.ensemble_m*np.asarray(ages)); target=(np.stack(active)*(w/w.sum())[:,None]).sum(0).astype(np.float32)
                target=clip_action(target,self.follower.get_state().astype(np.float32),self.cfg.max_step_deg)
                if self.last is not None: target=self.cfg.smooth_alpha*target+(1-self.cfg.smooth_alpha)*self.last
                self.last=target; self.follower.set_target(target)
            time.sleep(interval)

class AsyncPolicyRunner:
    def __init__(self, *, backend, follower, head, side, signs, offsets, joint_min, joint_max, config):
        self.ring=ChunkRingBuffer(); self.frames=FrameBuffer(); self.producer=_Producer(backend,follower,head,side,self.ring,self.frames,signs,offsets,joint_min,joint_max,config); self.consumer=None if config.dry_run else _Consumer(follower,self.ring,self.producer,config)
    def start(self): self.producer.start(); self.consumer and self.consumer.start(); return self
    def stop(self, join_timeout=2):
        self.producer.stop(); self.consumer and self.consumer.stop(); self.ring.clear(); self.producer.join(join_timeout); self.consumer and self.consumer.join(join_timeout)
    def latest_frames(self): return self.frames.get()
    def __enter__(self): return self.start()
    def __exit__(self,*_): self.stop(); return False
