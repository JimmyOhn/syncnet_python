#!/usr/bin/python
#-*- coding: utf-8 -*-

import time
import sys
import os

# 모듈 import
sys.path.append('/home/jimmyohn/work/opensource/syncnet_python')
from RealtimeSyncMonitor import RealtimeSyncMonitor

def test_comprehensive_sync_monitoring():
    """종합적인 RealtimeSyncMonitor 테스트 - 여러 비디오 파일"""
    
    print("🧪 === 종합적인 RealtimeSyncMonitor 테스트 ===")
    
    # 테스트할 비디오 파일들
    test_videos = [
        {
            'file': 'data/example.avi',
            'description': '기본 예제 비디오 (짧은 비디오)',
            'expected': '동기화 상태 불명 (더미 오디오)'
        },
        {
            'file': 'news_audio_delay_360p_500ms.mp4', 
            'description': '뉴스 비디오 500ms 오디오 지연',
            'expected': '불일치 (약 12-13프레임 지연)'
        }
    ]
    
    # 콜백 함수 정의
    def on_sync_change(status, offset, confidence):
        print(f"🔄 [콜백] 동기화 상태 변경: {status} (오프셋: {offset:.0f}프레임, 신뢰도: {confidence:.2f})")
    
    def on_sync_alert(offset, confidence):
        print(f"🚨 [알림] 동기화 불일치! 오프셋: {offset:.0f}프레임 ({offset*40:.0f}ms), 신뢰도: {confidence:.2f}")
    
    # 모델 로드 확인
    model_path = "data/syncnet_v2.model"
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일 없음: {model_path}")
        return
    
    # 각 비디오별 테스트
    for i, video_info in enumerate(test_videos, 1):
        video_file = video_info['file']
        
        print(f"\n🎬 === 테스트 {i}/{len(test_videos)}: {video_info['description']} ===")
        
        if not os.path.exists(video_file):
            print(f"❌ 비디오 파일 없음: {video_file}")
            continue
        
        print(f"📁 파일: {video_file}")
        print(f"🎯 예상 결과: {video_info['expected']}")
        
        # 비디오별 최적화된 설정
        if 'example.avi' in video_file:
            # 짧은 비디오용 설정
            monitor = RealtimeSyncMonitor(
                check_interval=5.0,
                min_frames=30,  # 더 관대한 기준
                buffer_duration=8.0
            )
            test_duration = 20  # 20초 테스트
        else:
            # 긴 비디오용 설정 (더 정확한 분석)
            monitor = RealtimeSyncMonitor(
                check_interval=8.0,
                min_frames=50,
                buffer_duration=12.0
            )
            test_duration = 30  # 30초 테스트
        
        # 모델 로드 및 콜백 설정
        monitor.load_model(model_path)
        monitor.set_callbacks(on_sync_change, on_sync_alert)
        
        print(f"⚙️  설정: 검사간격 {monitor.check_interval}초, 최소프레임 {monitor.min_frames}개")
        
        try:
            # 모니터링 시작
            video_thread, monitor_thread = monitor.start_monitoring(video_file)
            
            print(f"⏱️  {test_duration}초간 모니터링 시작...")
            
            # 주기적 상태 체크
            for t in range(test_duration):
                time.sleep(1)
                if t % 5 == 0:
                    status = monitor.get_status()
                    print(f"📊 [{t:2d}s] 상태: {status['status']}, 신뢰도: {status['confidence']:.2f}, 오프셋: {status['offset']:.0f}프레임")
        
        except KeyboardInterrupt:
            print("🛑 사용자 중단")
            break
        
        except Exception as e:
            print(f"❌ 테스트 오류: {e}")
        
        finally:
            monitor.stop_monitoring()
            
            # 개별 테스트 결과 요약
            final_status = monitor.get_status()
            print(f"\n📋 === 테스트 {i} 결과 요약 ===")
            print(f"🎯 최종 상태: {final_status['status']}")
            print(f"📊 최종 오프셋: {final_status['offset']:.0f}프레임 ({final_status['offset']*40:.0f}ms)")
            print(f"🎖️  최종 신뢰도: {final_status['confidence']:.2f}")
            
            # 안정성 분석
            if len(monitor.stability_buffer) >= 3:
                offsets = [r['offset'] for r in monitor.stability_buffer]
                confidences = [r['confidence'] for r in monitor.stability_buffer]
                import numpy as np
                print(f"📈 안정성 분석:")
                print(f"   - 오프셋 평균: {np.mean(offsets):.1f}프레임")
                print(f"   - 오프셋 편차: {np.std(offsets):.1f}프레임")
                print(f"   - 신뢰도 평균: {np.mean(confidences):.2f}")
            
            # 결과 평가
            if 'delay' in video_file:
                # 지연이 있는 비디오의 경우
                expected_offset_range = (10, 15)  # 500ms ≈ 12.5프레임
                if expected_offset_range[0] <= abs(final_status['offset']) <= expected_offset_range[1]:
                    print(f"✅ 예상 범위 내 결과: {abs(final_status['offset'])}프레임 (예상: {expected_offset_range[0]}-{expected_offset_range[1]}프레임)")
                else:
                    print(f"⚠️  예상 범위 벗어남: {abs(final_status['offset'])}프레임 (예상: {expected_offset_range[0]}-{expected_offset_range[1]}프레임)")
            
            print(f"✅ 테스트 {i} 완료\n")
            
            # 다음 테스트 전 잠시 대기
            if i < len(test_videos):
                print("⏳ 다음 테스트 준비 중...")
                time.sleep(3)
    
    print("🎉 === 모든 테스트 완료 ===")

def analyze_video_properties():
    """테스트 전 비디오 파일들의 속성 분석"""
    
    print("🔍 === 비디오 파일 속성 분석 ===")
    
    test_files = ['data/example.avi', 'news_audio_delay_360p_500ms.mp4']
    
    for video_file in test_files:
        if not os.path.exists(video_file):
            print(f"❌ 파일 없음: {video_file}")
            continue
        
        print(f"\n📁 분석 중: {video_file}")
        
        # ffprobe로 비디오 정보 확인
        import subprocess
        
        try:
            # 기본 정보
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                
                # 포맷 정보
                format_info = info.get('format', {})
                duration = float(format_info.get('duration', 0))
                
                print(f"   📊 길이: {duration:.1f}초")
                print(f"   📂 포맷: {format_info.get('format_name', 'Unknown')}")
                
                # 스트림 정보
                streams = info.get('streams', [])
                video_streams = [s for s in streams if s.get('codec_type') == 'video']
                audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
                
                print(f"   🎬 비디오 스트림: {len(video_streams)}개")
                if video_streams:
                    v = video_streams[0]
                    fps = eval(v.get('r_frame_rate', '25/1'))  # 분수 계산
                    print(f"      - 해상도: {v.get('width')}x{v.get('height')}")
                    print(f"      - FPS: {fps:.2f}")
                    print(f"      - 코덱: {v.get('codec_name')}")
                
                print(f"   🎵 오디오 스트림: {len(audio_streams)}개")
                if audio_streams:
                    a = audio_streams[0]
                    print(f"      - 샘플레이트: {a.get('sample_rate')}Hz")
                    print(f"      - 코덱: {a.get('codec_name')}")
                    print(f"      - 채널: {a.get('channels')}")
                
            else:
                print(f"   ❌ ffprobe 실패: {result.stderr}")
                
        except Exception as e:
            print(f"   ❌ 분석 오류: {e}")

if __name__ == "__main__":
    print("🚀 === RealtimeSyncMonitor 종합 테스트 시작 ===")
    
    # 1단계: 비디오 파일 속성 분석
    analyze_video_properties()
    
    print("\n" + "="*60)
    
    # 2단계: 실제 동기화 테스트
    test_comprehensive_sync_monitoring()
