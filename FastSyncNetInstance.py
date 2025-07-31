#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import numpy
import time, cv2, math
import python_speech_features
from scipy.io import wavfile
from SyncNetModel import *
from detectors import S3FD

class FastSyncNetInstance(torch.nn.Module):
    """
    Optimized SyncNet for fast face-based audio-video synchronization
    Only processes frames where faces are detected
    """
    
    def __init__(self, device='cpu', face_confidence=0.8, sample_interval=5):
        super(FastSyncNetInstance, self).__init__()
        
        self.device = device
        self.face_confidence = face_confidence
        self.sample_interval = sample_interval  # Process every N frames for speed
        
        # Load SyncNet model
        self.__S__ = S(num_layers_in_fc_layers=1024).to(self.device)
        
        # Set model to evaluation mode to fix BatchNorm issues with batch_size=1
        self.__S__.eval()
        
        # Load face detector
        self.face_detector = S3FD(device=self.device)
        
    def quick_evaluate(self, opt, videofile):
        """
        Fast evaluation using face detection and frame sampling
        """
        print('[INFO] Starting Fast SyncNet evaluation...')
        
        # Ensure model is in evaluation mode to avoid BatchNorm issues
        self.__S__.eval()
        
        # ========== ==========
        # Step 1: Extract audio (same as before)
        # ========== ==========
        
        # Extract audio first (lightweight operation)
        audio_path = self._extract_audio(videofile, opt.tmp_dir)
        sample_rate, audio = wavfile.read(audio_path)
        
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        
        print(f'[INFO] Audio processed: {len(audio)/16000:.2f}s')
        
        # ========== ==========
        # Step 2: Smart frame sampling with face detection
        # ========== ==========
        
        face_frames = self._detect_face_frames(videofile)
        
        if len(face_frames) == 0:
            print('[WARNING] No faces detected in video!')
            return None, None, None
            
        print(f'[INFO] Found {len(face_frames)} frames with faces (sampled every {self.sample_interval} frames)')
        
        # ========== ==========
        # Step 3: Process only face frames
        # ========== ==========
        
        selected_frames = self._select_representative_frames(face_frames, target_count=200)
        print(f'[INFO] Selected {len(selected_frames)} representative frames for processing')
        
        # Convert selected frames to tensor
        images = []
        frame_indices = []
        
        # Ensure we have enough frames for SyncNet processing (needs at least 5 frames)
        if len(selected_frames) < 10:
            print('[WARNING] Too few face frames detected. Using original approach with all frames.')
            return None, None, None
        
        for frame_info in selected_frames:
            frame_idx, face_region = frame_info['frame_idx'], frame_info['face_crop']
            
            # Resize to 224x224 for SyncNet
            face_resized = cv2.resize(face_region, (224, 224))
            images.append(face_resized)
            frame_indices.append(frame_idx)
        
        # Create video tensor with correct dimensions for Conv3d
        # Expected: [batch_size, channels, depth, height, width] = [1, 3, num_frames, 224, 224]
        im = numpy.stack(images, axis=0)  # [num_frames, height, width, channels]
        im = numpy.transpose(im, (3, 0, 1, 2))  # [channels, num_frames, height, width]
        im = numpy.expand_dims(im, axis=0)  # [1, channels, num_frames, height, width]
        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        print(f'[INFO] Video tensor shape: {imtv.shape}')  # Should be [1, 3, num_frames, 224, 224]
        
        # ========== ==========
        # Step 4: Fast SyncNet processing
        # ========== ==========
        
        lastframe = len(images) - 5
        im_feat = []
        cc_feat = []
        
        tS = time.time()
        
        # Use no_grad for inference to save memory and ensure eval mode
        with torch.no_grad():
            for i in range(0, lastframe, 1):
                if i % 50 == 0:
                    print(f'[INFO] Processing frame {i+1}/{lastframe}')
                
                # Process video frame (ensure correct dimensions for Conv3d)
                im_in = imtv[:, :, i:i+5, :, :]  # Shape: [1, 3, 5, 224, 224]
                im_out = self.__S__.forward_lip(im_in.to(self.device))
                im_feat.append(im_out.data.cpu())
                
                # Process corresponding audio frame (ensure correct dimensions)
                audio_frame_idx = frame_indices[i] if i < len(frame_indices) else i
                audio_start = audio_frame_idx * 4
                
                # Ensure audio slice is within bounds
                if audio_start + 20 <= cct.shape[3]:
                    cc_in = cct[:, :, :, audio_start:audio_start+20]  # Shape: [1, 1, 13, 20]
                    cc_out = self.__S__.forward_aud(cc_in.to(self.device))
                    cc_feat.append(cc_out.data.cpu())
                else:
                    # Pad or skip if audio is out of bounds
                    print(f'[WARNING] Audio frame {audio_frame_idx} out of bounds, skipping...')
                    break
        
        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)
        
        print(f'[INFO] Feature extraction completed in {time.time()-tS:.2f}s')
        
        # ========== ==========
        # Step 5: Compute offset (same as original)
        # ========== ==========
        
        from SyncNetInstance import calc_pdist
        
        dists = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1)
        
        minval, minidx = torch.min(mdist, 0)
        offset = opt.vshift - minidx
        conf = torch.median(mdist) - minval
        
        print(f'[RESULT] AV offset: {offset.item():.0f} frames')
        print(f'[RESULT] Confidence: {conf.item():.3f}')
        print(f'[RESULT] Min distance: {minval.item():.3f}')
        
        # Convert to time
        time_offset_ms = (offset.item() / 25) * 1000
        print(f'[RESULT] Time offset: {time_offset_ms:.0f}ms')
        
        dists_npy = numpy.array([dist.numpy() for dist in dists])
        return offset.numpy(), conf.numpy(), dists_npy
    
    def _extract_audio(self, videofile, tmp_dir):
        """Extract audio using ffmpeg"""
        import subprocess, os
        
        audio_path = os.path.join(tmp_dir, 'temp_audio.wav')
        os.makedirs(tmp_dir, exist_ok=True)
        
        command = f"ffmpeg -y -i {videofile} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_path}"
        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return audio_path
    
    def _detect_face_frames(self, videofile):
        """Detect faces in video frames with smart sampling"""
        print('[INFO] Detecting faces in video frames...')
        
        cap = cv2.VideoCapture(videofile)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        face_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at intervals for speed
            if frame_idx % self.sample_interval == 0:
                
                # Detect faces in this frame
                faces = self.face_detector.detect_faces(frame, conf_th=self.face_confidence)
                
                if len(faces) > 0:
                    # Get the largest face
                    largest_face = max(faces, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
                    
                    # Crop face region with some padding
                    x1, y1, x2, y2 = [int(x) for x in largest_face[:4]]
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.shape[0] > 50 and face_crop.shape[1] > 50:  # Minimum size check
                        face_frames.append({
                            'frame_idx': frame_idx,
                            'face_crop': face_crop,
                            'confidence': largest_face[4]
                        })
            
            frame_idx += 1
            
            if frame_idx % 500 == 0:
                print(f'[INFO] Processed {frame_idx}/{total_frames} frames, found {len(face_frames)} faces')
        
        cap.release()
        return face_frames
    
    def _select_representative_frames(self, face_frames, target_count=200):
        """Select representative frames to reduce computation"""
        if len(face_frames) <= target_count:
            return face_frames
        
        # Select frames with highest confidence and good temporal distribution
        face_frames.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Take top confidence frames but ensure temporal diversity
        selected = []
        used_indices = set()
        min_gap = max(1, len(face_frames) // target_count)
        
        for frame_info in face_frames:
            frame_idx = frame_info['frame_idx']
            
            # Check if this frame is too close to already selected frames
            too_close = any(abs(frame_idx - idx) < min_gap for idx in used_indices)
            
            if not too_close and len(selected) < target_count:
                selected.append(frame_info)
                used_indices.add(frame_idx)
        
        # Sort by frame index for processing
        selected.sort(key=lambda x: x['frame_idx'])
        return selected
    
    def loadParameters(self, path):
        """Load pre-trained SyncNet parameters"""
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
        self_state = self.__S__.state_dict()
        
        for name, param in loaded_state.items():
            self_state[name].copy_(param)
        
        # Ensure model is in evaluation mode after loading parameters
        self.__S__.eval()
        
        print(f"[INFO] Model loaded from {path} and set to evaluation mode")
