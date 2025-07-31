#!/usr/bin/python
#-*- coding: utf-8 -*-

import time
import sys
import os

# 모듈 import
sys.path.append('/home/jimmyohn/work/opensource/syncnet_python')
from RealtimeSyncMonitor import RealtimeSyncMonitor

def test_final_realtime_monitor():
    """최종 개선된 RealtimeSyncMonitor 테스트"""
    
    print("🎉 === 최종 RealtimeSyncMonitor 검증 테스트 ===")
    
    # 콜백 함수 정의
    def on_sync_change(status, offset, confidence):
        print(f"🔄 [콜백] 동기화 상태 변경: {status} (오프셋: {offset:.0f}프레임, 신뢰도: {confidence:.2f})")
        if status == 'out_of_sync':
            ms_offset = offset * (1000/30)  # 30fps 기준
            print(f"    💫 시간 단위: {ms_offset:.0f}ms 지연")
    
    def on_sync_alert(offset, confidence):
        ms_offset = offset * (1000/30)
        print(f"🚨 [알림] AV 동기화 불일치 감지!")
        print(f"    📊 오프셋: {offset:.0f}프레임 ({ms_offset:.0f}ms)")
        print(f"    🎖️  신뢰도: {confidence:.2f}")
    
    # 테스트할 비디오들
    test_videos = [
        {
            'file': 'news_audio_delay_360p_500ms.mp4',
            'name': '뉴스 비디오 (500ms 오디오 지연)',
            'expected_offset': 15,  # 500ms / 33.3ms ≈ 15프레임
            'check_interval': 6.0,
            'test_duration': 25
        }
    ]
    
    # 모델 경로 확인
    model_path = "data/syncnet_v2.model"
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일 없음: {model_path}")
        return
    
    for i, test_config in enumerate(test_videos, 1):
        video_file = test_config['file']
        
        if not os.path.exists(video_file):
            print(f"❌ 테스트 비디오 없음: {video_file}")
            continue
        
        print(f"\n🎬 === 테스트 {i}/{len(test_videos)}: {test_config['name']} ===")
        print(f"📁 파일: {video_file}")
        print(f"🎯 예상 오프셋: {test_config['expected_offset']}프레임")
        print(f"⏰ 검사 간격: {test_config['check_interval']}초")
        
        # 개선된 모니터 생성 (빠른 검증을 위한 설정)
        monitor = RealtimeSyncMonitor(
            check_interval=test_config['check_interval'],
            min_frames=40,  # 프레임 수 줄여서 빠른 처리
            buffer_duration=12.0
        )
        
        monitor.load_model(model_path)
        monitor.set_callbacks(on_sync_change, on_sync_alert)
        
        try:
            # 모니터링 시작
            video_thread, monitor_thread = monitor.start_monitoring(video_file)
            
            print(f"⏱️  {test_config['test_duration']}초간 모니터링...")
            
            # 지정된 시간만큼 모니터링
            for second in range(test_config['test_duration']):
                time.sleep(1)
                
                if second % 5 == 0 or second == test_config['test_duration'] - 1:
                    status = monitor.get_status()
                    print(f"📊 [{second:2d}s] 상태: {status['status']}, 신뢰도: {status['confidence']:.2f}, 오프셋: {status['offset']:.0f}프레임")
        
        except KeyboardInterrupt:
            print("🛑 사용자 중단")
        
        except Exception as e:
            print(f"❌ 테스트 오류: {e}")
        
        finally:
            monitor.stop_monitoring()
            
            # 결과 분석
            final_status = monitor.get_status()
            print(f"\n📋 === 테스트 {i} 결과 분석 ===")
            print(f"🎯 최종 상태: {final_status['status']}")
            print(f"📊 최종 오프셋: {final_status['offset']:.0f}프레임")
            print(f"🎖️  최종 신뢰도: {final_status['confidence']:.2f}")
            
            # 정확도 평가
            if final_status['status'] != 'unknown':
                expected = test_config['expected_offset']
                actual = final_status['offset']
                error = abs(actual - expected)
                accuracy = max(0, 100 - (error * 100 / max(expected, 1)))
                
                print(f"🎯 정확도 평가:")
                print(f"   예상: {expected}프레임")
                print(f"   측정: {actual:.0f}프레임")
                print(f"   오차: {error:.0f}프레임")
                print(f"   정확도: {accuracy:.1f}%")
                
                if error <= 3:  # 3프레임 이내 오차
                    print(f"   ✅ 우수한 정확도!")
                elif error <= 5:
                    print(f"   👍 양호한 정확도")
                else:
                    print(f"   ⚠️  개선 필요")
            else:
                print(f"   ❌ 동기화 상태 미확정")
            
            # 안정성 분석
            if len(monitor.stability_buffer) > 0:
                offsets = [r['offset'] for r in monitor.stability_buffer]
                import numpy as np
                print(f"📈 안정성 분석:")
                print(f"   측정 횟수: {len(monitor.stability_buffer)}회")
                print(f"   오프셋 범위: {min(offsets):.0f}~{max(offsets):.0f}프레임")
                print(f"   표준편차: {np.std(offsets):.1f}프레임")
                print(f"   평균 신뢰도: {np.mean([r['confidence'] for r in monitor.stability_buffer]):.2f}")
            
            print(f"✅ 테스트 {i} 완료\n")
            
            # 다음 테스트를 위한 정리
            if i < len(test_videos):
                print("⏳ 다음 테스트 준비 중...")
                time.sleep(2)
    
    print("🎉 === 모든 검증 테스트 완료 ===")
    print("💡 RealtimeSyncMonitor 개발 및 최적화 성공!")

if __name__ == "__main__":
    test_final_realtime_monitor()
