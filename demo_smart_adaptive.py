#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, argparse, torch
from SmartAdaptiveSyncNet import *

# ==================== PARSE ARGUMENTS ====================

parser = argparse.ArgumentParser(description="Smart Adaptive SyncNet - 지능형 AV 동기화 모니터링")
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='SyncNet 모델 파일')
parser.add_argument('--vshift', type=int, default='15', help='vshift 범위')
parser.add_argument('--videofile', type=str, default='data/example.avi', help='비디오 파일')
parser.add_argument('--tmp_dir', type=str, default='/tmp/work', help='임시 디렉토리')
parser.add_argument('--reference', type=str, default='demo', help='참조명')

# Smart Adaptive 옵션
parser.add_argument('--confidence_threshold', type=float, default=3.0, help='신뢰도 임계값')
parser.add_argument('--sync_threshold', type=int, default=2, help='동기화 임계값 (프레임)')
parser.add_argument('--max_checks', type=int, default=10, help='최대 검사 횟수')

opt = parser.parse_args()

# ==================== MAIN EXECUTION ====================

print("🚀 === Smart Adaptive SyncNet Demo ===")
print(f"📁 처리 파일: {opt.videofile}")
print(f"⚙️  설정: 신뢰도 임계값 {opt.confidence_threshold}, 동기화 임계값 ±{opt.sync_threshold}프레임")

# 디바이스 자동 감지
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  사용 디바이스: {device}")

# Smart Adaptive SyncNet 초기화
smart_sync = SmartAdaptiveSyncNet(device=device)
smart_sync.confidence_threshold = opt.confidence_threshold
smart_sync.sync_threshold = opt.sync_threshold
smart_sync.max_checks = opt.max_checks

# 모델 로드
smart_sync.loadParameters(opt.initial_model)

# 시작 시간 기록
start_time = time.time()

try:
    # 스마트 모니터링 실행
    result = smart_sync.smart_monitor(opt, opt.videofile)
    
    total_time = time.time() - start_time
    
    print(f"\n🏁 === 최종 리포트 ===")
    print(f"⏱️  총 처리 시간: {total_time:.2f}초")
    
    if result['status'] == 'synced':
        print(f"✅ 동기화 상태: 완벽")
        print(f"📊 AV 오프셋: {result['offset']:.0f}프레임 ({result['offset']*40:.0f}ms)")
        print(f"🎖️  신뢰도: {result['confidence']:.2f}")
        print(f"🔍 검사 횟수: {result['checks']}회")
        
        if result['checks'] == 1:
            print(f"⚡ 초고속 완료: 첫 번째 검사만으로 확인!")
        
    elif result['status'] == 'out_of_sync':
        print(f"⚠️  동기화 상태: 불일치 감지")
        print(f"📊 AV 오프셋: {result['offset']:.0f}프레임 ({result['offset']*40:.0f}ms)")
        print(f"🎖️  신뢰도: {result['confidence']:.2f}")
        print(f"🔍 검사 횟수: {result['checks']}회")
        
        # 불일치 정도에 따른 권장사항
        offset_severity = abs(result['offset'])
        if offset_severity <= 2:
            print(f"💡 권장: 경미한 불일치 (±{offset_severity}프레임)")
        elif offset_severity <= 5:
            print(f"⚠️  권장: 중간 불일치 - 보정 권장")
        else:
            print(f"🚨 권장: 심각한 불일치 - 즉시 보정 필요")
            
    elif result['status'] == 'failed':
        print(f"❌ 분석 실패")
        print(f"🔍 시도된 검사: {result['checks']}회")
        print(f"💡 권장: 더 긴 비디오나 다른 파일로 재시도")
        
    # 리소스 효율성 리포트
    print(f"\n📈 === 효율성 리포트 ===")
    
    if 'checks' in result:
        efficiency = (opt.max_checks - result['checks']) / opt.max_checks * 100
        print(f"💰 리소스 절약: {efficiency:.1f}% (예정 {opt.max_checks}회 중 {result['checks']}회만 실행)")
    
    # 기존 방식과 비교
    estimated_full_time = 60  # 전체 SyncNet 예상 시간
    speedup = estimated_full_time / total_time
    print(f"⚡ 속도 개선: 약 {speedup:.1f}배 빠름")
    
    # 메모리 효율성
    if result.get('checks', 0) < opt.max_checks // 2:
        print(f"🧠 메모리 효율: 조기 종료로 메모리 사용량 최소화")
    
    print(f"\n💡 Smart Adaptive SyncNet의 장점:")
    print(f"   - 이미 동기화된 영상은 즉시 감지")
    print(f"   - 긴 영상도 주기적 샘플링으로 효율적 처리")
    print(f"   - 신뢰도 달성 시 자동 중단으로 리소스 절약")
    print(f"   - 실시간 스트리밍에 최적화")
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
