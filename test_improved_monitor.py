#!/usr/bin/python
#-*- coding: utf-8 -*-

import time
import sys
import os

# 모듈 import
sys.path.append('/home/jimmyohn/work/opensource/syncnet_python')
from RealtimeSyncMonitor import RealtimeSyncMonitor

def test_improved_realtime_monitor():
    """개선된 RealtimeSyncMonitor 테스트"""
    
    print("🧪 === 개선된 RealtimeSyncMonitor 안정성 테스트 ===")
    
    # 콜백 함수 정의
    def on_sync_change(status, offset, confidence):
        print(f"🔄 [콜백] 동기화 상태 변경: {status} (오프셋: {offset:.0f}프레임, 신뢰도: {confidence:.2f})")
    
    def on_sync_alert(offset, confidence):
        print(f"🚨 [알림] 동기화 불일치! 오프셋: {offset:.0f}프레임 ({offset*40:.0f}ms), 신뢰도: {confidence:.2f}")
    
    # 개선된 실시간 모니터 생성 (더 엄격한 설정)
    monitor = RealtimeSyncMonitor(
        check_interval=8.0,  # 8초마다 검사 (더 긴 간격)
        min_frames=50,       # 더 많은 프레임 요구
        buffer_duration=15.0 # 더 긴 버퍼링
    )
    
    # 모델 로드
    model_path = "data/syncnet_v2.model"
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일 없음: {model_path}")
        return
    
    monitor.load_model(model_path)
    monitor.set_callbacks(on_sync_change, on_sync_alert)
    
    # 테스트용 비디오
    test_video = "data/example.avi"
    
    if not os.path.exists(test_video):
        print(f"❌ 테스트 비디오 없음: {test_video}")
        print("📁 사용 가능한 파일들:")
        for f in os.listdir("data"):
            if f.endswith(('.mp4', '.avi', '.mov')):
                print(f"   - {f}")
        return
    
    print(f"🎬 테스트 비디오: {test_video}")
    print(f"📊 설정: 검사간격 8초, 최소프레임 50개, 버퍼 15초")
    print(f"🎯 목표: 안정적이고 일관된 동기화 결과 확보")
    
    # 모니터링 시작
    try:
        video_thread, monitor_thread = monitor.start_monitoring(test_video)
        
        # 40초간 모니터링 (5번의 검사 예상)
        for i in range(40):
            time.sleep(1)
            if i % 5 == 0:
                status = monitor.get_status()
                print(f"📊 [{i:2d}s] 상태: {status['status']}, 신뢰도: {status['confidence']:.2f}, 오프셋: {status['offset']:.0f}프레임")
    
    except KeyboardInterrupt:
        print("🛑 사용자 중단")
    
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
    
    finally:
        monitor.stop_monitoring()
        
        # 최종 결과 요약
        final_status = monitor.get_status()
        print(f"\n📋 === 최종 결과 요약 ===")
        print(f"🎯 최종 상태: {final_status['status']}")
        print(f"📊 최종 오프셋: {final_status['offset']:.0f}프레임 ({final_status['offset']*40:.0f}ms)")
        print(f"🎖️  최종 신뢰도: {final_status['confidence']:.2f}")
        print(f"⏱️  안정성 버퍼 크기: {len(monitor.stability_buffer)}개")
        
        if len(monitor.stability_buffer) >= 3:
            offsets = [r['offset'] for r in monitor.stability_buffer]
            import numpy as np
            print(f"📈 오프셋 안정성: 평균 {np.mean(offsets):.1f}, 편차 {np.std(offsets):.1f}")
        
        print("✅ 개선된 테스트 완료")

if __name__ == "__main__":
    test_improved_realtime_monitor()
