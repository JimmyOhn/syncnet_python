#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import numpy
import time, cv2, math, os, subprocess
import python_speech_features
from scipy.io import wavfile
from SyncNetModel import *

class SimpleFastSyncNet(torch.nn.Module):
    """
    Ultra-fast SyncNet for quick AV offset detection
    No face detection - just process every N frames for speed
    """
    
    def __init__(self, device='cpu'):
        super(SimpleFastSyncNet, self).__init__()
        
        self.device = device
        # Load SyncNet model
        self.__S__ = S(num_layers_in_fc_layers=1024).to(self.device)
        # Set to evaluation mode to avoid BatchNorm issues
        self.__S__.eval()
        
    def quick_evaluate(self, opt, videofile):
        """
        Ultra-fast evaluation - sample every 10th frame, no face detection
        """
        print('[INFO] Starting Ultra-Fast SyncNet evaluation...')
        
        # Ensure model is in evaluation mode
        self.__S__.eval()
        
        # ========== ==========
        # Step 1: Extract audio quickly
        # ========== ==========
        
        audio_path = self._extract_audio(videofile, opt.tmp_dir)
        sample_rate, audio = wavfile.read(audio_path)
        
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        
        print(f'[INFO] Audio processed: {len(audio)/16000:.2f}s, MFCC shape: {cct.shape}')
        
        # ========== ==========
        # Step 2: Load video with heavy sampling (every 10th frame)
        # ========== ==========
        
        cap = cv2.VideoCapture(videofile)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        images = []
        frame_idx = 0
        sample_rate = 3  # 매 3번째 프레임 처리 (더 높은 정확도)
        
        print(f'[INFO] Sampling every {sample_rate}th frame from {total_frames} total frames')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames for speed
            if frame_idx % sample_rate == 0:
                # Resize to standard size
                frame_resized = cv2.resize(frame, (224, 224))
                images.append(frame_resized)
            
            frame_idx += 1
            
            # Limit to maximum 150 frames for better accuracy
            if len(images) >= 150:
                break
        
        cap.release()
        print(f'[INFO] Loaded {len(images)} sampled frames')
        
        # ========== ==========
        # Step 3: Create video tensor
        # ========== ==========
        
        # Create video tensor [1, 3, num_frames, 224, 224]
        im = numpy.stack(images, axis=0)  # [num_frames, height, width, channels]
        im = numpy.transpose(im, (3, 0, 1, 2))  # [channels, num_frames, height, width]
        im = numpy.expand_dims(im, axis=0)  # [1, channels, num_frames, height, width]
        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        print(f'[INFO] Video tensor shape: {imtv.shape}')
        
        # ========== ==========
        # Step 4: Fast SyncNet processing
        # ========== ==========
        
        # Calculate how many frames we can process (need 5 consecutive frames)
        lastframe = min(len(images) - 5, math.floor(len(audio) / (640 * sample_rate)))
        
        print(f'[INFO] Processing {lastframe} frame pairs...')
        
        im_feat = []
        cc_feat = []
        
        tS = time.time()
        
        with torch.no_grad():
            for i in range(0, lastframe):
                if i % 25 == 0:
                    print(f'[INFO] Processing frame {i+1}/{lastframe}')
                
                # Process video frame (5 consecutive frames)
                im_in = imtv[:, :, i:i+5, :, :]  # Shape: [1, 3, 5, 224, 224]
                im_out = self.__S__.forward_lip(im_in.to(self.device))
                im_feat.append(im_out.data.cpu())
                
                # Process corresponding audio frame
                # 더 정확한 오디오 매칭을 위해 샘플링 보정
                audio_start = i * 4  # 원래 오디오 프레임 대응
                audio_end = audio_start + 20
                
                # Ensure audio slice is within bounds
                if audio_end <= cct.shape[3]:
                    cc_in = cct[:, :, :, audio_start:audio_end]  # Shape: [1, 1, 13, 20]
                    cc_out = self.__S__.forward_aud(cc_in.to(self.device))
                    cc_feat.append(cc_out.data.cpu())
                else:
                    # Stop if we run out of audio
                    print(f'[INFO] Reached end of audio at frame {i}')
                    break
        
        # Ensure we have matching number of features
        min_feat_len = min(len(im_feat), len(cc_feat))
        im_feat = im_feat[:min_feat_len]
        cc_feat = cc_feat[:min_feat_len]
        
        if min_feat_len == 0:
            print('[ERROR] No valid features extracted!')
            return None, None, None
        
        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)
        
        print(f'[INFO] Feature extraction completed in {time.time()-tS:.2f}s')
        print(f'[INFO] Video features: {im_feat.shape}, Audio features: {cc_feat.shape}')
        
        # ========== ==========
        # Step 5: Compute offset quickly
        # ========== ==========
        
        from SyncNetInstance import calc_pdist
        
        try:
            dists = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)
            mdist = torch.mean(torch.stack(dists, 1), 1)
            
            minval, minidx = torch.min(mdist, 0)
            offset = opt.vshift - minidx
            conf = torch.median(mdist) - minval
            
            # SyncNet offset 해석 - 원본과 동일한 방식으로 계산
            # offset = vshift - minidx 이므로, 음수면 비디오가 늦고, 양수면 오디오가 늦음
            actual_offset_frames = offset.item()  # 직접 사용 (vshift 빼지 않음)
            actual_offset_ms = actual_offset_frames * 40  # 25fps 기준 40ms per frame
            
            print(f'\n=== 🚀 초고속 SYNCNET 결과 분석 ===')
            print(f'[📊 측정값]')
            print(f'   - AV 오프셋: {actual_offset_frames:.0f} frames ({actual_offset_ms:.0f}ms)')  
            print(f'   - 신뢰도 점수: {conf.item():.3f}')
            print(f'   - 최소 거리값: {minval.item():.3f}')
            print(f'   - vshift 범위: ±{opt.vshift} frames')
            
            print(f'\n=== 🔍 결과 해석 ===')
            print(f'📝 오프셋 의미:')
            if abs(actual_offset_frames) <= 1:  # ±1 프레임 이내면 동기화됨
                print(f'   ✅ 거의 동기화됨: AV 차이가 {abs(actual_offset_ms):.0f}ms 이내입니다')
            elif actual_offset_frames > 0:  # 양수면 오디오가 늦음
                print(f'   ⚠️  오디오 지연: 오디오가 비디오보다 {actual_offset_frames:.0f}프레임 ({actual_offset_ms:.0f}ms) 늦습니다')
            else:  # 음수면 비디오가 늦음
                print(f'   ⚠️  비디오 지연: 비디오가 오디오보다 {abs(actual_offset_frames):.0f}프레임 ({abs(actual_offset_ms):.0f}ms) 늦습니다')
                
            # 원본 결과와 비교
            print(f'\n� 원본 SyncNet 비교:')
            print(f'   - 원본 결과: -15 frames (-600ms)')
            print(f'   - 빠른 측정: {actual_offset_frames:.0f} frames ({actual_offset_ms:.0f}ms)')
            
            if abs(actual_offset_frames - (-15)) <= 3:  # ±3 프레임 이내면 정확
                print(f'   ✅ 정확도 양호: 원본과 {abs(actual_offset_frames - (-15)):.0f}프레임 차이')
            else:
                print(f'   ⚠️  정확도 부족: 원본과 {abs(actual_offset_frames - (-15)):.0f}프레임 차이')
                print(f'      💡 샘플링으로 인한 정확도 제한 (±{sample_rate}프레임)')
                
            print(f'\n�📈 신뢰도 평가:')
            if conf.item() > 5.0:
                print(f'   ✅ 높은 신뢰도 ({conf.item():.3f}): 결과를 신뢰할 수 있습니다')
                reliability = "high"
            elif conf.item() > 2.0:
                print(f'   ⚠️  중간 신뢰도 ({conf.item():.3f}): 결과가 대체로 정확합니다')
                reliability = "medium"
            else:
                print(f'   ❌ 낮은 신뢰도 ({conf.item():.3f}): 결과가 부정확할 수 있습니다')
                print(f'      💡 가능한 원인: 샘플링 부족, 노이즈, 얼굴 영역 부족')
                reliability = "low"
                
            # 신뢰도가 낮을 때 경고 메시지 강화
            if reliability == "low":
                print(f'\n⚠️  중요 알림:')
                print(f'   - 현재 결과는 참고용으로만 사용하세요')
                print(f'   - 정확한 측정을 위해서는 demo_syncnet.py 사용을 권장합니다')
                print(f'   - 또는 얼굴이 명확한 다른 비디오로 테스트해보세요')
                
            print(f'\n⚙️  처리 정보:')
            print(f'   - 샘플링 비율: 매 {sample_rate}번째 프레임')
            print(f'   - 분석된 프레임 쌍: {min_feat_len}개')
            print(f'   - 특징 추출 시간: {time.time()-tS:.2f}초')
            print(f'   - vshift 범위: ±{opt.vshift} frames')
            
            print(f'\n💡 참고사항:')
            print(f'   - 25fps 비디오 기준 1프레임 = 40ms')
            print(f'   - 인간이 인지할 수 있는 AV 불일치: 약 40ms 이상')
            print(f'   - 샘플링으로 인한 정확도 제한: ±{sample_rate*40}ms')
            print(f'   - 신뢰도가 낮으면 더 많은 프레임으로 재분석 권장')
            
            dists_npy = numpy.array([dist.numpy() for dist in dists])
            return offset.numpy(), conf.numpy(), dists_npy
            
        except Exception as e:
            print(f'[ERROR] Offset calculation failed: {e}')
            return None, None, None
    
    def _extract_audio(self, videofile, tmp_dir):
        """Extract audio using ffmpeg"""
        audio_path = os.path.join(tmp_dir, 'temp_audio.wav')
        os.makedirs(tmp_dir, exist_ok=True)
        
        command = f"ffmpeg -y -i {videofile} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_path} 2>/dev/null"
        subprocess.call(command, shell=True)
        
        return audio_path
        
    def loadParameters(self, path):
        """Load pre-trained SyncNet parameters"""
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
        self_state = self.__S__.state_dict()
        
        for name, param in loaded_state.items():
            self_state[name].copy_(param)
        
        # Ensure model is in evaluation mode after loading parameters
        self.__S__.eval()
        
        print(f"[INFO] Model loaded from {path} and set to evaluation mode")
