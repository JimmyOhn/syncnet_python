#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, argparse
import torch
from FastSyncNetInstance import FastSyncNetInstance

# ==================== LOAD PARAMS ====================

parser = argparse.ArgumentParser(description="Fast SyncNet - Face-based optimization")

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='Pre-trained SyncNet model')
parser.add_argument('--vshift', type=int, default='15', help='Maximum frame shift to search')
parser.add_argument('--videofile', type=str, default="data/example.avi", help='Input video file')
parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='Temporary directory')
parser.add_argument('--reference', type=str, default="fast_demo", help='Reference name')
parser.add_argument('--face_confidence', type=float, default=0.8, help='Face detection confidence threshold')
parser.add_argument('--sample_interval', type=int, default=5, help='Process every N frames for speed')

opt = parser.parse_args()

# ==================== RUN FAST EVALUATION ====================

print("=== FAST SYNCNET DEMO ===")
print(f"Video: {opt.videofile}")
print(f"Face confidence: {opt.face_confidence}")
print(f"Sample interval: {opt.sample_interval}")
print()

# Check device availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[INFO] Using device: {device}')

# Initialize Fast SyncNet
start_time = time.time()

s = FastSyncNetInstance(
    device=device,
    face_confidence=opt.face_confidence,
    sample_interval=opt.sample_interval
)

s.loadParameters(opt.initial_model)

# Run fast evaluation
try:
    offset, conf, dists = s.quick_evaluate(opt, opt.videofile)
    
    if offset is not None:
        total_time = time.time() - start_time
        
        print()
        print("=== RESULTS ===")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"AV offset: {offset.item():.0f} frames")
        print(f"Time offset: {(offset.item()/25)*1000:.0f} ms")
        print(f"Confidence: {conf.item():.3f}")
        print()
        print("=== SPEED COMPARISON ===")
        print("Traditional SyncNet: ~90+ seconds")
        print(f"Fast SyncNet: {total_time:.2f} seconds")
        print(f"Speed improvement: {90/total_time:.1f}x faster!")
        
    else:
        print("[ERROR] Could not process video - no faces detected")
        
except Exception as e:
    print(f"[ERROR] Processing failed: {e}")
    import traceback
    traceback.print_exc()
