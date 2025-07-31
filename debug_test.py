#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, argparse, torch, os
from SmartAdaptiveSyncNet import *

# 간단한 디버그 테스트
print("🔧 === Smart Adaptive SyncNet 디버그 테스트 ===")

videofile = "news_audio_delay_360p_500ms.mp4"
model_path = "data/syncnet_v2.model"

# 파일 존재 확인
if not os.path.exists(videofile):
    print(f"❌ 비디오 파일 없음: {videofile}")
    exit(1)

if not os.path.exists(model_path):
    print(f"❌ 모델 파일 없음: {model_path}")
    exit(1)

print(f"✅ 파일 확인 완료")

# 디바이스 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  디바이스: {device}")

# Smart Adaptive SyncNet 초기화
try:
    smart_sync = SmartAdaptiveSyncNet(device=device)
    print(f"✅ SmartAdaptiveSyncNet 초기화 완료")
except Exception as e:
    print(f"❌ 초기화 실패: {e}")
    exit(1)

# 모델 로드
try:
    smart_sync.loadParameters(model_path)
    print(f"✅ 모델 로드 완료")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit(1)

# 옵션 설정
class SimpleOpt:
    def __init__(self):
        self.tmp_dir = '/tmp/syncnet_test'
        self.vshift = 15
        self.reference = 'test'

opt = SimpleOpt()

print(f"\n🚀 빠른 사전 검사 시작...")

try:
    result = smart_sync._quick_precheck(videofile, opt)
    
    if result is None:
        print(f"❌ 사전 검사 실패")
    else:
        offset, confidence, is_synced = result
        print(f"✅ 사전 검사 성공!")
        print(f"   오프셋: {offset:.0f} 프레임")
        print(f"   신뢰도: {confidence:.3f}")
        print(f"   동기화: {is_synced}")
        
except Exception as e:
    print(f"❌ 사전 검사 예외: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🏁 디버그 테스트 완료")
