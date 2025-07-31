#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import numpy
import time, cv2, threading, queue
from SmartAdaptiveSyncNet import SmartAdaptiveSyncNet

class RealtimeSyncMonitor:
    """
    실시간 AV 동기화 모니터
    - 스트리밍 데이터 실시간 수신
    - 주기적 동기화 검사
    - 자동 알림 시스템
    """
    
    def __init__(self, device='cpu', check_interval=10.0, min_frames=30, buffer_duration=10.0):
        self.device = device
        self.check_interval = check_interval  # 검사 간격 (초)
        self.min_frames = min_frames  # 최소 필요 프레임 수
        self.buffer_duration = buffer_duration  # 버퍼링 지속 시간 (초)
        
        # Smart Adaptive SyncNet 인스턴스
        self.sync_net = SmartAdaptiveSyncNet(device=device)
        
        # 모니터링 상태
        self.is_monitoring = False
        self.last_check_time = 0
        self.sync_status = 'unknown'
        self.confidence = 0.0
        self.offset = 0.0
        
        # 데이터 버퍼 (더 큰 크기와 더 긴 보관 시간)
        self.video_buffer = queue.Queue(maxsize=3000)  # 훨씬 더 큰 버퍼
        self.audio_buffer = queue.Queue(maxsize=3000)
        self.frame_collection_duration = 12.0  # 12초간 프레임 수집 (더 긴 구간)
        
        # 적응형 샘플링
        self.sampling_interval = 2  # 시작: 매 2번째 프레임
        self.insufficient_frames_count = 0  # 프레임 부족 횟수
        
        # 안정성 개선을 위한 추가 설정
        self.analysis_window_size = 150  # 분석용 프레임 수 (6초 분량)
        self.stability_buffer = []  # 최근 결과들의 안정성 확인용
        self.min_confidence = 3.0  # 최소 신뢰도 임계값
        self.original_audio_source = None  # 원본 오디오 소스 저장
        
        # 콜백 함수들
        self.sync_callback = None  # 동기화 상태 변경 시 호출
        self.alert_callback = None  # 불일치 감지 시 호출
    
    def load_model(self, model_path):
        """SyncNet 모델 로드"""
        self.sync_net.loadParameters(model_path)
        print(f"🎯 실시간 모니터 준비 완료: {model_path}")
    
    def set_callbacks(self, sync_callback=None, alert_callback=None):
        """콜백 함수 설정"""
        self.sync_callback = sync_callback
        self.alert_callback = alert_callback
    
    def start_monitoring(self, video_source, audio_source=None):
        """실시간 모니터링 시작"""
        print(f"🚀 실시간 AV 동기화 모니터링 시작")
        print(f"📊 검사 간격: {self.check_interval}초")
        print(f"🎯 최소 필요 프레임: {self.min_frames}개")
        print(f"⏱️  프레임 수집 범위: {self.frame_collection_duration}초")
        
        # 원본 오디오 소스 저장 (실제 오디오 사용을 위해)
        self.original_audio_source = video_source
        self.is_monitoring = True
        
        # 비디오 스트림 스레드
        video_thread = threading.Thread(
            target=self._video_stream_worker, 
            args=(video_source,)
        )
        video_thread.daemon = True
        video_thread.start()
        
        # 모니터링 스레드  
        monitor_thread = threading.Thread(target=self._monitor_worker)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print(f"✅ 실시간 모니터링 활성화됨")
        
        return video_thread, monitor_thread
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        print(f"🛑 실시간 모니터링 중지됨")
    
    def _video_stream_worker(self, video_source):
        """비디오 스트림 처리 워커 - 반복 재생 지원"""
        frame_count = 0
        consecutive_failures = 0
        
        while self.is_monitoring:
            cap = cv2.VideoCapture(video_source)
            
            if not cap.isOpened():
                print(f"   ❌ 비디오 열기 실패: {video_source}")
                time.sleep(1)
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            duration = total_frames / fps
            
            print(f"   🎬 비디오 정보: {total_frames}프레임, {duration:.1f}초, {fps}fps")
            
            video_frame_count = 0
            
            while self.is_monitoring and video_frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        print(f"   ⚠️  연속 프레임 읽기 실패, 비디오 재시작")
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0  # 성공 시 리셋
                
                # 적응형 프레임 버퍼링 (샘플링 간격 조정)
                if frame_count % self.sampling_interval == 0:
                    timestamp = time.time()
                    frame_data = {
                        'frame': cv2.resize(frame, (224, 224)),
                        'timestamp': timestamp,
                        'frame_id': frame_count,
                        'video_frame_id': video_frame_count
                    }
                    
                    try:
                        self.video_buffer.put_nowait(frame_data)
                    except queue.Full:
                        # 버퍼가 가득 찬 경우 오래된 데이터 제거
                        try:
                            self.video_buffer.get_nowait()
                            self.video_buffer.put_nowait(frame_data)
                        except queue.Empty:
                            pass
                
                frame_count += 1
                video_frame_count += 1
                time.sleep(0.04)  # 25fps 시뮬레이션
            
            cap.release()
            
            # 짧은 비디오인 경우 반복을 위해 잠시 대기
            if duration < 30 and self.is_monitoring:
                print(f"   🔄 짧은 비디오 ({duration:.1f}초) 반복 재생")
                time.sleep(0.5)  # 짧은 대기 후 재시작
    
    def _monitor_worker(self):
        """동기화 모니터링 워커"""
        # 초기 버퍼링 대기
        print(f"⏳ 초기 프레임 버퍼링 중... ({self.frame_collection_duration}초 대기)")
        time.sleep(self.frame_collection_duration + 3)  # 버퍼 시간 + 3초 여유
        
        while self.is_monitoring:
            current_time = time.time()
            
            # 검사 간격 확인
            if current_time - self.last_check_time >= self.check_interval:
                self._perform_sync_check()
                self.last_check_time = current_time
            
            time.sleep(1.0)  # 1초마다 확인
    
    def _perform_sync_check(self):
        """동기화 검사 수행"""
        print(f"\n🔍 [{time.strftime('%H:%M:%S')}] 동기화 검사 중...")
        
        # 버퍼에서 최근 프레임들 수집
        frames = self._collect_recent_frames()
        
        print(f"   📊 수집된 프레임: {len(frames)}개")
        
        if len(frames) < self.min_frames:
            print(f"   ⚠️  충분한 프레임 없음 ({len(frames)}개 < {self.min_frames}개)")
            self.insufficient_frames_count += 1
            
            # 적응형 샘플링: 프레임이 부족하면 더 조밀하게 수집
            if self.insufficient_frames_count >= 2 and self.sampling_interval > 1:
                self.sampling_interval = max(1, self.sampling_interval - 1)
                print(f"   🔧 샘플링 간격 조정: 매 {self.sampling_interval}번째 프레임")
                self.insufficient_frames_count = 0
            
            print(f"   💡 더 많은 프레임 수집을 위해 대기 중...")
            return
        else:
            # 충분한 프레임이 수집되면 샘플링 간격 복구
            if self.insufficient_frames_count > 0:
                self.insufficient_frames_count = 0
                if self.sampling_interval < 2:
                    self.sampling_interval = 2
                    print(f"   ✅ 샘플링 간격 복구: 매 {self.sampling_interval}번째 프레임")
        
        try:
            # 임시 비디오 파일 생성
            temp_video = self._create_temp_video(frames)
            
            if temp_video:
                # 빠른 동기화 검사
                result = self._quick_sync_analysis(temp_video)
                
                if result:
                    old_status = self.sync_status
                    candidate_offset = result['offset']
                    candidate_confidence = result['confidence']
                    
                    # 결과 안정성 검증
                    stable_result = self._validate_result_stability(candidate_offset, candidate_confidence)
                    
                    if stable_result:
                        self.offset = stable_result['offset']
                        self.confidence = stable_result['confidence']
                        
                        # 동기화 상태 판단 (더 엄격한 기준)
                        if abs(self.offset) <= 2 and self.confidence > self.min_confidence:
                            self.sync_status = 'synced'
                        else:
                            self.sync_status = 'out_of_sync'
                        
                        print(f"   📊 안정화된 결과: 오프셋 {self.offset:.0f}프레임, 신뢰도 {self.confidence:.2f}")
                        print(f"   🎯 상태: {self.sync_status}")
                        
                        # 상태 변경 시 콜백 호출
                        if old_status != self.sync_status:
                            if self.sync_callback:
                                self.sync_callback(self.sync_status, self.offset, self.confidence)
                            
                            # 불일치 감지 시 알림
                            if self.sync_status == 'out_of_sync' and self.alert_callback:
                                self.alert_callback(self.offset, self.confidence)
                        
                        # 높은 신뢰도로 동기화됨이 확인되면 검사 간격 증가
                        if self.sync_status == 'synced' and self.confidence > 5.0:
                            self.check_interval = min(30.0, self.check_interval * 1.5)
                            print(f"   💡 검사 간격 연장: {self.check_interval:.1f}초")
                    else:
                        print(f"   ⚠️  결과가 불안정하여 대기 중... (후보: 오프셋 {candidate_offset:.0f}, 신뢰도 {candidate_confidence:.2f})")
                        
                else:
                    print(f"   ❌ 동기화 분석 실패")
                
                # 임시 파일 정리
                import os
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                    
        except Exception as e:
            print(f"   ❌ 검사 오류: {e}")
    
    def _collect_recent_frames(self):
        """최근 프레임들 수집 - 비파괴적 복사 방식"""
        current_time = time.time()
        
        # 큐의 내용을 비파괴적으로 복사
        temp_frames = []
        frames_to_restore = []
        
        # 1단계: 큐에서 모든 프레임을 임시로 수집하면서 복사본 저장
        while not self.video_buffer.empty():
            try:
                frame_data = self.video_buffer.get_nowait()
                temp_frames.append(frame_data)
                # 유효한 프레임은 다시 큐에 넣을 준비
                if current_time - frame_data['timestamp'] <= self.frame_collection_duration * 1.5:  # 1.5배 여유
                    frames_to_restore.append(frame_data)
            except queue.Empty:
                break
        
        # 2단계: 유효한 프레임들을 큐에 복원
        for frame_data in frames_to_restore:
            try:
                self.video_buffer.put_nowait(frame_data)
            except queue.Full:
                # 큐가 가득 차면 오래된 프레임 하나 제거 후 재시도
                try:
                    self.video_buffer.get_nowait()
                    self.video_buffer.put_nowait(frame_data)
                except queue.Empty:
                    pass
        
        # 3단계: 분석용 프레임 필터링 (원본 보존하면서)
        recent_frames = []
        for frame_data in temp_frames:
            if current_time - frame_data['timestamp'] <= self.frame_collection_duration:
                # 프레임 데이터 복사 (원본 보존)
                frame_copy = {
                    'frame': frame_data['frame'].copy(),
                    'timestamp': frame_data['timestamp'],
                    'frame_id': frame_data['frame_id']
                }
                recent_frames.append(frame_copy)
        
        # 시간순 정렬
        recent_frames.sort(key=lambda x: x['timestamp'])
        
        # 적응형 프레임 수 조정
        target_frames = max(self.min_frames, 80)  # 최소 50~80프레임 목표
        
        if len(recent_frames) > target_frames:
            # 너무 많으면 균등하게 샘플링
            step = len(recent_frames) // target_frames
            recent_frames = recent_frames[::max(1, step)][:target_frames]
        
        buffer_status = f"버퍼: {self.video_buffer.qsize()}/{self.video_buffer.maxsize}"
        
        print(f"   🔍 수집 시간 범위: {self.frame_collection_duration}초 (샘플링: 매 {self.sampling_interval}번째)")
        print(f"   📦 임시 수집: {len(temp_frames)}개, 유효 프레임: {len(recent_frames)}개, {buffer_status}")
        
        # 프레임 부족 시 경고 및 조치
        if len(recent_frames) < self.min_frames:
            shortage = self.min_frames - len(recent_frames)
            print(f"   ⚠️  목표 프레임 수 미달: {len(recent_frames)}/{self.min_frames}개 (부족: {shortage}개)")
            
            # 프레임 수집 전략 조정
            if self.frame_collection_duration < 20:
                self.frame_collection_duration = min(20, self.frame_collection_duration + 2)
                print(f"   🔧 수집 범위 확장: {self.frame_collection_duration}초")
        
        return recent_frames
    
    def _validate_result_stability(self, offset, confidence):
        """결과 안정성 검증 - 최근 결과들과 비교하여 일관성 확인"""
        result = {'offset': offset, 'confidence': confidence, 'timestamp': time.time()}
        self.stability_buffer.append(result)
        
        # 최근 5개 결과만 유지
        if len(self.stability_buffer) > 5:
            self.stability_buffer.pop(0)
        
        # 적응형 안정성 검증: 2개 또는 3개 결과로 검증
        required_samples = 2 if confidence > 2.0 else 3  # 신뢰도가 높으면 2개만으로도 충분
        
        if len(self.stability_buffer) < required_samples:
            print(f"   🔄 안정성 검증: 데이터 부족 ({len(self.stability_buffer)}/{required_samples})")
            return None
        
        recent_results = self.stability_buffer[-required_samples:]  # 최근 N개
        offsets = [r['offset'] for r in recent_results]
        confidences = [r['confidence'] for r in recent_results]
        
        # 오프셋 안정성 검사 (표준편차 기준 완화)
        import numpy as np
        offset_std = np.std(offsets)
        confidence_mean = np.mean(confidences)
        offset_mean = np.mean(offsets)
        
        print(f"   📈 안정성 검증: 오프셋 편차 {offset_std:.1f}, 평균 신뢰도 {confidence_mean:.2f}")
        print(f"   📊 최근 {required_samples}개 결과: {[f'{o:.0f}' for o in offsets]} (평균: {offset_mean:.1f})")
        
        # 개선된 안정성 기준
        max_offset_std = 7.0 if confidence_mean < 2.0 else 5.0  # 신뢰도에 따른 적응형 기준
        min_confidence = 1.0  # 최소 신뢰도 완화
        
        if offset_std <= max_offset_std and confidence_mean >= min_confidence:
            # 안정된 결과로 인정, 평균값 사용
            stable_offset = round(offset_mean)
            stable_confidence = confidence_mean
            
            print(f"   ✅ 결과 안정화 완료: 오프셋 {offset_mean:.1f} → {stable_offset}프레임")
            print(f"   🎯 안정성 기준: 편차 {offset_std:.1f} ≤ {max_offset_std}, 신뢰도 {confidence_mean:.2f} ≥ {min_confidence}")
            
            return {
                'offset': stable_offset,
                'confidence': stable_confidence
            }
        else:
            print(f"   ⏳ 결과 불안정: 편차 {offset_std:.1f} > {max_offset_std} 또는 신뢰도 {confidence_mean:.2f} < {min_confidence}")
            
            # 특별 경우: 연속으로 같은 오프셋이 나오면 즉시 승인
            if len(set(offsets)) == 1 and confidence_mean > 1.5:
                print(f"   🚀 특별 승인: 동일 오프셋 연속 감지 ({offsets[0]:.0f}프레임)")
                return {
                    'offset': round(offsets[0]),
                    'confidence': confidence_mean
                }
            
            return None
    
    def _create_temp_video(self, frames):
        """임시 비디오 파일 생성 (더미 오디오 포함)"""
        if not frames:
            return None
        
        temp_video_only = f"/tmp/temp_video_only_{int(time.time())}.avi"
        temp_video_with_audio = f"/tmp/temp_sync_check_{int(time.time())}.mp4"
        
        try:
            # 1단계: 비디오만 생성
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_video_only, fourcc, 25.0, (224, 224))
            
            for frame_data in frames:
                out.write(frame_data['frame'])
            
            out.release()
            
            # 2단계: 더미 오디오 트랙 추가 (ffmpeg 사용)
            import subprocess
            import os
            
            duration = len(frames) / 25.0  # 25fps 기준 길이
            
            # 무음 오디오 생성 및 비디오와 합성
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'quiet',
                '-i', temp_video_only,
                '-f', 'lavfi', '-i', f'anullsrc=channel_layout=mono:sample_rate=16000',
                '-t', str(duration),
                '-c:v', 'copy', '-c:a', 'aac',
                '-shortest',
                temp_video_with_audio
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 임시 비디오 파일 정리
            if os.path.exists(temp_video_only):
                os.remove(temp_video_only)
            
            if result.returncode == 0 and os.path.exists(temp_video_with_audio):
                return temp_video_with_audio
            else:
                print(f"   ⚠️  더미 오디오 추가 실패, 비디오만 사용: {result.stderr}")
                # 실패 시 원본 비디오 반환 (SmartAdaptiveSyncNet에서 예외 처리)
                return temp_video_only
            
        except Exception as e:
            print(f"   ❌ 임시 비디오 생성 실패: {e}")
            # 정리
            import os
            if os.path.exists(temp_video_only):
                os.remove(temp_video_only)
            if os.path.exists(temp_video_with_audio):
                os.remove(temp_video_with_audio)
            return None
    
    def _quick_sync_analysis(self, video_path):
        """개선된 동기화 분석 - 실제 오디오 사용"""
        try:
            # 간단한 옵션 객체 생성
            class SimpleOpt:
                def __init__(self):
                    self.tmp_dir = '/tmp'
                    self.vshift = 15
            
            opt = SimpleOpt()
            
            # 실제 원본 비디오의 오디오를 사용하도록 수정
            if self.original_audio_source and self.original_audio_source != video_path:
                print(f"   🎵 원본 비디오에서 오디오 추출: {self.original_audio_source}")
                
                # 원본 비디오에서 오디오 추출
                original_audio_path = self.sync_net._extract_audio(self.original_audio_source, opt.tmp_dir)
                
                if original_audio_path:
                    # 임시로 SmartAdaptiveSyncNet의 오디오 추출 메서드를 오버라이드
                    original_extract = self.sync_net._extract_audio
                    
                    def use_original_audio(videofile, tmp_dir):
                        return original_audio_path
                    
                    self.sync_net._extract_audio = use_original_audio
                    
                    try:
                        # 빠른 사전 검사 수행
                        result = self.sync_net._quick_precheck(video_path, opt)
                    finally:
                        # 원래 메서드 복원
                        self.sync_net._extract_audio = original_extract
                else:
                    print(f"   ⚠️  원본 오디오 추출 실패, 더미 오디오 사용")
                    result = self.sync_net._quick_precheck(video_path, opt)
            else:
                # 일반적인 경우 (더미 오디오 사용)
                result = self.sync_net._quick_precheck(video_path, opt)
            
            if result:
                offset, confidence, is_synced = result
                return {'offset': offset, 'confidence': confidence, 'is_synced': is_synced}
            
            return None
            
        except Exception as e:
            print(f"   ❌ 빠른 분석 실패: {e}")
            return None
    
    def get_status(self):
        """현재 동기화 상태 반환"""
        return {
            'status': self.sync_status,
            'offset': self.offset,
            'confidence': self.confidence,
            'is_monitoring': self.is_monitoring,
            'last_check': self.last_check_time,
            'check_interval': self.check_interval
        }

# 사용 예제
if __name__ == "__main__":
    # 콜백 함수 정의
    def on_sync_change(status, offset, confidence):
        print(f"🔄 동기화 상태 변경: {status} (오프셋: {offset:.0f}프레임, 신뢰도: {confidence:.2f})")
    
    def on_sync_alert(offset, confidence):
        print(f"🚨 동기화 불일치 알림! 오프셋: {offset:.0f}프레임 ({offset*40:.0f}ms)")
    
    # 실시간 모니터 생성
    monitor = RealtimeSyncMonitor(check_interval=5.0)
    monitor.load_model("data/syncnet_v2.model")
    monitor.set_callbacks(on_sync_change, on_sync_alert)
    
    # 모니터링 시작
    monitor.start_monitoring("news_audio_delay_360p_500ms.mp4")
    
    try:
        # 30초간 모니터링
        for i in range(30):
            time.sleep(1)
            if i % 5 == 0:
                status = monitor.get_status()
                print(f"📊 현재 상태: {status['status']}, 신뢰도: {status['confidence']:.2f}")
    
    except KeyboardInterrupt:
        print("사용자 중단")
    
    finally:
        monitor.stop_monitoring()
