#!/usr/bin/python
#-*- coding: utf-8 -*-

import time
import sys
import os

# 모듈 import
sys.path.append('/home/jimmyohn/work/opensource/syncnet_python')
from RealtimeSyncMonitor import RealtimeSyncMonitor

def test_realtime_monitor():
    """RealtimeSyncMonitor 테스트 (오디오 없는 임시 비디오 처리)"""
    
    print("🧪 === RealtimeSyncMonitor 오디오 없는 비디오 처리 테스트 ===")
    
    # 콜백 함수 정의
    def on_sync_change(status, offset, confidence):
        print(f"🔄 동기화 상태 변경: {status} (오프셋: {offset:.0f}프레임, 신뢰도: {confidence:.2f})")
    
    def on_sync_alert(offset, confidence):
        print(f"🚨 동기화 불일치 알림! 오프셋: {offset:.0f}프레임 ({offset*40:.0f}ms)")
    
    # 실시간 모니터 생성
    monitor = RealtimeSyncMonitor(check_interval=3.0)  # 3초마다 검사
    
    # 모델 로드
    model_path = "data/syncnet_v2.model"
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일 없음: {model_path}")
        return
    
    monitor.load_model(model_path)
    monitor.set_callbacks(on_sync_change, on_sync_alert)
    
    # 테스트용 비디오 (실제 비디오 파일)
    test_video = "data/example.avi"  # 또는 다른 비디오 파일
    
    if not os.path.exists(test_video):
        print(f"❌ 테스트 비디오 없음: {test_video}")
        print("📁 사용 가능한 파일들:")
        for f in os.listdir("data"):
            if f.endswith(('.mp4', '.avi', '.mov')):
                print(f"   - {f}")
        return
    
    print(f"🎬 테스트 비디오: {test_video}")
    
    # 모니터링 시작
    try:
        video_thread, monitor_thread = monitor.start_monitoring(test_video)
        
        # 15초간 모니터링
        for i in range(15):
            time.sleep(1)
            if i % 3 == 0:
                status = monitor.get_status()
                print(f"📊 [{i:2d}s] 상태: {status['status']}, 신뢰도: {status['confidence']:.2f}, 오프셋: {status['offset']:.0f}")
    
    except KeyboardInterrupt:
        print("🛑 사용자 중단")
    
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
    
    finally:
        monitor.stop_monitoring()
        print("✅ 테스트 완료")

if __name__ == "__main__":
    test_realtime_monitor()
