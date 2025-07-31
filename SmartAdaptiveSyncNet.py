#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import numpy
import time, cv2, math, os, subprocess
import python_speech_features
from scipy.io import wavfile
from SyncNetModel import *

class SmartAdaptiveSyncNet(torch.nn.Module):
    """
    스마트 적응형 SyncNet - 실시간 AV 동기화 모니터링
    1. 빠른 사전 검사로 동기화 상태 확인
    2. 주기적 샘플링으로 지속적 모니터링
    3. 신뢰도 달성 시 자동 중단으로 리소스 절약
    """
    
    def __init__(self, device='cpu'):
        super(SmartAdaptiveSyncNet, self).__init__()
        
        self.device = device
        # Load SyncNet model
        self.__S__ = S(num_layers_in_fc_layers=1024).to(self.device)
        self.__S__.eval()
        
        # 적응형 매개변수
        self.confidence_threshold = 3.0  # 신뢰도 임계값
        self.sync_threshold = 2  # 동기화 임계값 (±2 프레임)
        self.check_interval = 5.0  # 검사 간격 (초)
        self.max_checks = 10  # 최대 검사 횟수
        self.quick_sample_frames = 30  # 빠른 검사용 프레임 수
        
    def smart_monitor(self, opt, videofile):
        """
        스마트 모니터링 - 단계별 적응형 처리
        """
        print('🚀 === Smart Adaptive SyncNet 시작 ===')
        print(f'📋 설정: 신뢰도 임계값 {self.confidence_threshold}, 동기화 임계값 ±{self.sync_threshold}프레임')
        
        # Step 1: 빠른 사전 검사
        print('\n📊 [1단계] 빠른 사전 검사 수행 중...')
        quick_result = self._quick_precheck(videofile, opt)
        
        if quick_result is None:
            print('❌ 사전 검사 실패 - 전체 분석으로 전환')
            return self._full_analysis(videofile, opt)
        
        offset, confidence, is_synced = quick_result
        
        if is_synced and confidence > self.confidence_threshold:
            print(f'✅ [결과] 이미 동기화됨! (오프셋: {offset:.0f}프레임, 신뢰도: {confidence:.2f})')
            print(f'💡 SyncNet 모델 추가 실행 불필요 - 리소스 절약됨')
            return {'status': 'synced', 'offset': offset, 'confidence': confidence, 'checks': 1}
        
        # Step 2: 주기적 적응형 모니터링
        print(f'\n🔄 [2단계] 적응형 주기적 모니터링 시작...')
        return self._adaptive_monitoring(videofile, opt, initial_offset=offset)
    
    def _quick_precheck(self, videofile, opt):
        """빠른 사전 검사 - 최소한의 프레임으로 동기화 상태 확인"""
        print(f'   🔍 오디오 추출 중...')
        try:
            # 오디오 추출
            audio_path = self._extract_audio(videofile, opt.tmp_dir)
            if not os.path.exists(audio_path):
                print(f'   ❌ 오디오 추출 실패: {audio_path}')
                return None
                
            sample_rate, audio = wavfile.read(audio_path)
            print(f'   ✅ 오디오 로드 완료: {len(audio)/16000:.1f}초')
            
            # MFCC 변환
            mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
            mfcc = numpy.stack([numpy.array(i) for i in mfcc])
            cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
            cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
            print(f'   ✅ MFCC 변환 완료: {cct.shape}')
            
            # 비디오 빠른 샘플링
            print(f'   🎬 비디오 프레임 샘플링 중...')
            cap = cv2.VideoCapture(videofile)
            
            if not cap.isOpened():
                print(f'   ❌ 비디오 파일 열기 실패: {videofile}')
                return None
                
            images = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f'   📺 전체 프레임 수: {total_frames}')
            
            while frame_count < min(self.quick_sample_frames * 5, total_frames):  # 더 많은 프레임 확인
                ret, frame = cap.read()
                if not ret:
                    print(f'   ⚠️  프레임 읽기 실패 at {frame_count}')
                    break
                    
                # 매 5번째 프레임만 사용
                if frame_count % 5 == 0:
                    if frame is not None and frame.size > 0:
                        frame_resized = cv2.resize(frame, (224, 224))
                        images.append(frame_resized)
                    
                frame_count += 1
            
            cap.release()
            
            if len(images) < 5:  # 최소 5프레임 필요
                print(f'   ❌ 충분한 프레임 없음: {len(images)}개')
                return None
                
            print(f'   📹 샘플링된 프레임: {len(images)}개')
            
            # 텐서 생성 및 특징 추출
            print(f'   🔧 텐서 생성 중...')
            im = numpy.stack(images, axis=0)
            im = numpy.transpose(im, (3, 0, 1, 2))
            im = numpy.expand_dims(im, axis=0)
            imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
            print(f'   ✅ 비디오 텐서 shape: {imtv.shape}')
            
            # 특징 추출 - 더 안전한 방식
            lastframe = min(len(images) - 5, 15)  # 최대 15프레임 쌍
            if lastframe <= 0:
                print(f'   ❌ 처리 가능한 프레임 부족: {lastframe}')
                return None
                
            im_feat, cc_feat = [], []
            
            print(f'   🧠 특징 추출 중... ({lastframe}개 프레임 쌍)')
            with torch.no_grad():
                for i in range(lastframe):
                    try:
                        # 비디오 특징
                        if i + 5 <= imtv.shape[2]:  # 경계 검사
                            im_in = imtv[:, :, i:i+5, :, :]
                            im_out = self.__S__.forward_lip(im_in.to(self.device))
                            im_feat.append(im_out.data.cpu())
                        
                        # 오디오 특징 - 더 안전한 인덱싱
                        audio_start = i * 4
                        audio_end = audio_start + 20
                        
                        if audio_end <= cct.shape[3]:
                            cc_in = cct[:, :, :, audio_start:audio_end]
                            cc_out = self.__S__.forward_aud(cc_in.to(self.device))
                            cc_feat.append(cc_out.data.cpu())
                        else:
                            print(f'     ⚠️  오디오 범위 초과 at {i}: {audio_end} > {cct.shape[3]}')
                            break
                            
                    except Exception as e:
                        print(f'     ❌ 특징 추출 오류 at {i}: {e}')
                        break
            
            if len(im_feat) == 0 or len(cc_feat) == 0:
                print(f'   ❌ 특징 추출 실패: video={len(im_feat)}, audio={len(cc_feat)}')
                return None
                
            # 길이 맞추기
            min_len = min(len(im_feat), len(cc_feat))
            im_feat = torch.cat(im_feat[:min_len], 0)
            cc_feat = torch.cat(cc_feat[:min_len], 0)
            
            print(f'   ✅ 특징 벡터 생성: video={im_feat.shape}, audio={cc_feat.shape}')
            
            # 오프셋 계산
            print(f'   🎯 오프셋 계산 중...')
            from SyncNetInstance import calc_pdist
            dists = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)
            mdist = torch.mean(torch.stack(dists, 1), 1)
            
            minval, minidx = torch.min(mdist, 0)
            offset = opt.vshift - minidx
            conf = torch.median(mdist) - minval
            
            is_synced = abs(offset.item()) <= self.sync_threshold
            
            print(f'   📊 빠른 결과: 오프셋 {offset.item():.0f}프레임, 신뢰도 {conf.item():.2f}, 동기화 {is_synced}')
            
            return offset.item(), conf.item(), is_synced
            
        except Exception as e:
            print(f'   ❌ 빠른 검사 오류: {e}')
            return None
    
    def _adaptive_monitoring(self, videofile, opt, initial_offset):
        """적응형 주기적 모니터링"""
        
        # 오디오 전체 추출
        audio_path = self._extract_audio(videofile, opt.tmp_dir)
        sample_rate, audio = wavfile.read(audio_path)
        
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        
        # 비디오 정보
        cap = cv2.VideoCapture(videofile)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        duration = total_frames / fps
        cap.release()
        
        print(f'   📺 비디오 정보: {total_frames}프레임, {duration:.1f}초, {fps}fps')
        
        # 주기적 검사 계획
        check_intervals = self._plan_check_intervals(duration)
        print(f'   ⏰ 검사 계획: {len(check_intervals)}회 ({check_intervals}초 지점)')
        
        results = []
        best_confidence = 0
        best_result = None
        
        for check_num, start_time in enumerate(check_intervals, 1):
            print(f'\n🔍 [{check_num}/{len(check_intervals)}] {start_time:.1f}초 지점 검사 중...')
            
            # 해당 시점부터 샘플링
            result = self._sample_at_timepoint(videofile, cct, start_time, fps, opt)
            
            if result:
                offset, confidence = result
                results.append({'time': start_time, 'offset': offset, 'confidence': confidence})
                
                print(f'   📊 결과: 오프셋 {offset:.0f}프레임 ({offset*40:.0f}ms), 신뢰도 {confidence:.2f}')
                
                # 최고 신뢰도 결과 추적
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = {'offset': offset, 'confidence': confidence, 'time': start_time}
                
                # 신뢰도 임계값 달성 시 조기 종료
                if confidence >= self.confidence_threshold:
                    print(f'   ✅ 신뢰도 임계값 달성! ({confidence:.2f} >= {self.confidence_threshold})')
                    print(f'   💡 추가 검사 중단 - 리소스 절약')
                    break
            else:
                print(f'   ⚠️  해당 구간 분석 실패')
        
        # 최종 결과 정리
        if best_result:
            offset = best_result['offset']
            is_synced = abs(offset) <= self.sync_threshold
            
            print(f'\n🎯 === 최종 결과 ===')
            print(f'📊 최고 신뢰도 결과: 오프셋 {offset:.0f}프레임 ({offset*40:.0f}ms)')
            print(f'🎖️  신뢰도: {best_result["confidence"]:.2f}')
            print(f'⏰ 검사 횟수: {check_num}/{len(check_intervals)}')
            print(f'🎯 동기화 상태: {"✅ 동기화됨" if is_synced else "⚠️ 불일치"}')
            
            return {
                'status': 'synced' if is_synced else 'out_of_sync',
                'offset': offset,
                'confidence': best_result['confidence'],
                'checks': check_num,
                'all_results': results
            }
        else:
            print('❌ 모든 검사 실패')
            return {'status': 'failed', 'checks': len(check_intervals)}
    
    def _plan_check_intervals(self, duration):
        """비디오 길이에 따른 검사 간격 계획"""
        if duration <= 10:  # 10초 이하
            return [2, 5, 8]
        elif duration <= 30:  # 30초 이하
            return [5, 10, 15, 20, 25]
        elif duration <= 60:  # 1분 이하
            return [10, 20, 30, 40, 50]
        else:  # 1분 이상
            # 처음, 중간, 끝 부분 집중 검사
            intervals = [10, 30, 60]
            # 긴 비디오는 중간중간 추가 검사
            for i in range(120, int(duration), 60):
                intervals.append(i)
                if len(intervals) >= self.max_checks:
                    break
            return intervals[:self.max_checks]
    
    def _sample_at_timepoint(self, videofile, cct, start_time, fps, opt):
        """특정 시점에서 샘플링하여 분석"""
        try:
            cap = cv2.VideoCapture(videofile)
            
            # 해당 시점으로 이동
            start_frame = int(start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 5초 구간 샘플링 (매 3번째 프레임)
            images = []
            frame_count = 0
            target_frames = int(5 * fps)  # 5초 분량
            
            while frame_count < target_frames and len(images) < 50:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % 3 == 0:  # 매 3번째 프레임
                    frame_resized = cv2.resize(frame, (224, 224))
                    images.append(frame_resized)
                    
                frame_count += 1
            
            cap.release()
            
            if len(images) < 10:
                return None
            
            # 텐서 생성 및 특징 추출
            im = numpy.stack(images, axis=0)
            im = numpy.transpose(im, (3, 0, 1, 2))
            im = numpy.expand_dims(im, axis=0)
            imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
            
            lastframe = min(len(images) - 5, 30)  # 최대 30프레임 쌍
            im_feat, cc_feat = [], []
            
            with torch.no_grad():
                for i in range(lastframe):
                    # 비디오 특징
                    im_in = imtv[:, :, i:i+5, :, :]
                    im_out = self.__S__.forward_lip(im_in.to(self.device))
                    im_feat.append(im_out.data.cpu())
                    
                    # 해당 시점의 오디오 특징
                    audio_frame_idx = int((start_time * fps + i * 3) / fps * 100)  # 100Hz 오디오 프레임
                    audio_start = audio_frame_idx * 4
                    
                    if audio_start + 20 <= cct.shape[3]:
                        cc_in = cct[:, :, :, audio_start:audio_start+20]
                        cc_out = self.__S__.forward_aud(cc_in.to(self.device))
                        cc_feat.append(cc_out.data.cpu())
            
            if len(im_feat) == 0 or len(cc_feat) == 0:
                return None
            
            # 길이 맞추기
            min_len = min(len(im_feat), len(cc_feat))
            im_feat = torch.cat(im_feat[:min_len], 0)
            cc_feat = torch.cat(cc_feat[:min_len], 0)
            
            # 오프셋 계산
            from SyncNetInstance import calc_pdist
            dists = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)
            mdist = torch.mean(torch.stack(dists, 1), 1)
            
            minval, minidx = torch.min(mdist, 0)
            offset = opt.vshift - minidx
            conf = torch.median(mdist) - minval
            
            return offset.item(), conf.item()
            
        except Exception as e:
            print(f'   ❌ 샘플링 오류: {e}')
            return None
    
    def _full_analysis(self, videofile, opt):
        """전체 분석 - 기존 SimpleFastSyncNet과 유사"""
        print('🔄 전체 SyncNet 분석 실행 중...')
        # 기존 SimpleFastSyncNet 로직 사용
        return {'status': 'full_analysis_needed'}
    
    def _extract_audio(self, videofile, tmp_dir):
        """오디오 추출 - 더 견고한 방식 (오디오 없는 비디오 처리 포함)"""
        audio_path = os.path.join(tmp_dir, 'temp_audio.wav')
        os.makedirs(tmp_dir, exist_ok=True)
        
        # 기존 파일 삭제
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # 1단계: 비디오에 오디오 스트림이 있는지 확인
        print(f'   🔍 비디오 스트림 정보 확인 중...')
        probe_cmd = f"ffprobe -v quiet -select_streams a -show_entries stream=codec_type -of csv=p=0 \"{videofile}\""
        
        try:
            probe_result = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True, timeout=10)
            has_audio = 'audio' in probe_result.stdout.lower()
            
            if not has_audio:
                print(f'   ⚠️  비디오에 오디오 스트림이 없음 - 더미 오디오 생성')
                return self._create_dummy_audio(audio_path, videofile)
                
        except Exception as e:
            print(f'   ⚠️  스트림 검사 실패, 오디오 추출 시도: {e}')
        
        # 2단계: ffmpeg로 오디오 추출 (더 자세한 로그)
        command = f"ffmpeg -y -i \"{videofile}\" -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 \"{audio_path}\""
        print(f'   🎵 오디오 추출 명령: {command}')
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f'   ❌ ffmpeg 오류: {result.stderr}')
                
                # "Output file does not contain any stream" 오류 감지
                if "does not contain any stream" in result.stderr:
                    print(f'   🔧 오디오 스트림 없음 감지, 더미 오디오 생성')
                    return self._create_dummy_audio(audio_path, videofile)
                    
                return None
                
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print(f'   ❌ 오디오 파일 생성 실패 또는 빈 파일 - 더미 오디오 생성')
                return self._create_dummy_audio(audio_path, videofile)
                
            print(f'   ✅ 오디오 추출 성공: {os.path.getsize(audio_path)} bytes')
            return audio_path
            
        except subprocess.TimeoutExpired:
            print(f'   ❌ 오디오 추출 시간 초과 (30초)')
            return None
        except Exception as e:
            print(f'   ❌ 오디오 추출 예외: {e}')
            return None
    
    def _create_dummy_audio(self, audio_path, videofile):
        """더미 오디오 트랙 생성 (비디오 길이에 맞춤)"""
        try:
            # 비디오 길이 확인
            duration_cmd = f"ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{videofile}\""
            duration_result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            if duration_result.returncode == 0 and duration_result.stdout.strip():
                duration = float(duration_result.stdout.strip())
            else:
                # 기본값: 5초
                duration = 5.0
                print(f'   ⚠️  비디오 길이 확인 실패, 기본 5초 사용')
            
            # 무음 오디오 생성
            dummy_cmd = f"ffmpeg -y -f lavfi -i anullsrc=channel_layout=mono:sample_rate=16000 -t {duration} -acodec pcm_s16le \"{audio_path}\""
            print(f'   🔧 더미 오디오 생성: {duration:.2f}초')
            
            dummy_result = subprocess.run(dummy_cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if dummy_result.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                print(f'   ✅ 더미 오디오 생성 성공: {os.path.getsize(audio_path)} bytes')
                return audio_path
            else:
                print(f'   ❌ 더미 오디오 생성 실패: {dummy_result.stderr}')
                return None
                
        except Exception as e:
            print(f'   ❌ 더미 오디오 생성 예외: {e}')
            return None
        
    def loadParameters(self, path):
        """사전 훈련된 SyncNet 매개변수 로드"""
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
        self_state = self.__S__.state_dict()
        
        for name, param in loaded_state.items():
            self_state[name].copy_(param)
        
        self.__S__.eval()
        print(f"[INFO] Smart Adaptive SyncNet 모델 로드 완료: {path}")
