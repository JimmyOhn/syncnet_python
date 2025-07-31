#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, os, math, glob, torch, cv2
from SimpleFastSyncNet import *

# ==================== PARSE ARGUMENTS ====================

parser = argparse.ArgumentParser(description = "FaceSync")
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='')
parser.add_argument('--batch_size', type=int, default='20', help='')
parser.add_argument('--vshift', type=int, default='15', help='')
parser.add_argument('--videofile', type=str, default='data/example.avi', help='')
parser.add_argument('--tmp_dir', type=str, default='/tmp/work', help='')
parser.add_argument('--reference', type=str, default='demo', help='')
opt = parser.parse_args()

# ==================== LOAD MODEL AND DATA ====================

print("=== Ultra-Fast SyncNet Demo ===")
print(f"Processing: {opt.videofile}")

# Auto-detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

s = SimpleFastSyncNet(device=device)

# ==================== LOAD PRETRAINED MODEL ====================

s.loadParameters(opt.initial_model)
print("Model %s loaded."%opt.initial_model)

# ==================== GET OFFSETS ====================

start_time = time.time()

try:
    offset, conf, dists = s.quick_evaluate(opt, opt.videofile)
    
    if offset is not None:
        total_time = time.time() - start_time
        # 원본과 동일한 방식으로 오프셋 계산
        actual_offset_frames = offset
        actual_offset_ms = actual_offset_frames * 40
        
        print(f"\n=== 🏁 최종 요약 ===")
        print(f"⏱️  총 처리 시간: {total_time:.2f}초 (원본: 99.83초)")
        
        sync_status = "동기화됨" if abs(actual_offset_frames) <= 1 else "불일치 감지"
        print(f"🎯 AV 동기화 상태: {sync_status}")
        
        # 파일명에서 예상 지연시간 확인 (예: 500ms)
        if '500ms' in opt.videofile:
            expected_delay = 500
            original_result = -600  # 원본 SyncNet 결과
            
            print(f"📁 파일명 기준 예상 지연: {expected_delay}ms")
            print(f"🔍 원본 SyncNet 결과: {original_result}ms")
            print(f"⚡ 빠른 측정 결과: {actual_offset_ms:.0f}ms")
            
            accuracy_vs_original = abs(original_result - actual_offset_ms)
            print(f"📊 원본 대비 정확도: {accuracy_vs_original:.0f}ms 차이")
            
            if accuracy_vs_original < 200:
                print(f"✅ 빠른 측정 정확도 양호 (200ms 이내 차이)")
                print(f"💨 속도 개선: {99.83/total_time:.1f}배 빠름")
            else:
                print(f"⚠️  빠른 측정 정확도 부족 - 신뢰도가 낮거나 샘플링 부족")
                print(f"💡 정확한 결과가 필요하면 원본 demo_syncnet.py 사용 권장")
        
    else:
        print("[❌ ERROR] AV 오프셋 계산에 실패했습니다")
        
except Exception as e:
    print(f"[❌ ERROR] 처리 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
