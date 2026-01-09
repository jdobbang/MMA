#!/usr/bin/env python3
"""
Migrate SMPL Directories
========================

smpl 파라미터 디렉토리를 /workspace/dataset에서 /workspace/MMA/dataset로 이동

Features:
- Dry-run 모드 (기본값, 실제 변경 없음)
- 포괄적인 사전 검증 (pre-flight validation)
- 매니페스트 파일 생성으로 추적 가능
- 이동 후 검증
- 상세한 로깅 및 에러 처리

Usage:
    # Dry-run 모드 (안전한 미리보기)
    python migrate_smpl_directories.py

    # 사전 검증만 실행
    python migrate_smpl_directories.py --verify-only

    # 실제 이동 실행
    python migrate_smpl_directories.py --execute

    # 커스텀 경로
    python migrate_smpl_directories.py \\
        --source-root /workspace/dataset \\
        --target-root /workspace/MMA/dataset \\
        --execute
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


# =============================================================================
# 상수 정의
# =============================================================================

DEFAULT_SOURCE_ROOT = "/workspace/dataset"
DEFAULT_TARGET_ROOT = "/workspace/MMA/dataset"
DEFAULT_DATASETS = ['03_grappling2', '13_mma2']


# =============================================================================
# 유틸리티 함수
# =============================================================================

def format_bytes(bytes_value: int) -> str:
    """
    바이트를 읽기 쉬운 형식으로 변환

    Args:
        bytes_value: 바이트 값

    Returns:
        str: 포맷된 문자열 (예: "1.2 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def get_directory_size(path: str) -> int:
    """
    디렉토리의 전체 크기 계산

    Args:
        path: 디렉토리 경로

    Returns:
        int: 바이트 단위 총 크기
    """
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total += os.path.getsize(filepath)
    except Exception:
        pass
    return total


def count_npy_files(path: str) -> int:
    """
    디렉토리의 .npy 파일 개수 카운트

    Args:
        path: 디렉토리 경로

    Returns:
        int: .npy 파일 개수
    """
    count = 0
    try:
        for item in os.listdir(path):
            if item.endswith('.npy'):
                count += 1
    except Exception:
        pass
    return count


# =============================================================================
# Discovery 함수
# =============================================================================

def discover_smpl_directories(
    source_root: str,
    target_root: str,
    datasets: List[str]
) -> List[Dict[str, str]]:
    """
    마이그레이션이 필요한 모든 SMPL 디렉토리 검색

    Args:
        source_root: 소스 루트 디렉토리
        target_root: 타겟 루트 디렉토리
        datasets: 처리할 데이터셋 리스트

    Returns:
        List[Dict]: 마이그레이션 정보 딕셔너리 리스트
            - source: 소스 경로
            - target: 타겟 경로
            - target_parent: 타겟 부모 디렉토리
            - dataset: 데이터셋 이름
            - session: 세션 이름
    """
    migrations = []

    for dataset in datasets:
        dataset_path = Path(source_root) / dataset

        if not dataset_path.exists():
            continue

        # 모든 세션 디렉토리 순회
        for session_dir in sorted(dataset_path.iterdir()):
            if not session_dir.is_dir():
                continue

            source_smpl = session_dir / 'processed_data' / 'smpl'

            if not source_smpl.exists():
                continue

            # 타겟 경로 생성
            target_processed_data = Path(target_root) / dataset / session_dir.name / 'processed_data'
            target_smpl = target_processed_data / 'smpl'

            migrations.append({
                'source': str(source_smpl),
                'target': str(target_smpl),
                'target_parent': str(target_processed_data),
                'dataset': dataset,
                'session': session_dir.name
            })

    return migrations


# =============================================================================
# 검증 함수
# =============================================================================

def verify_preconditions(migrations: List[Dict]) -> Tuple[bool, List[str]]:
    """
    안전한 마이그레이션을 위한 사전 조건 검증

    Args:
        migrations: 마이그레이션 정보 리스트

    Returns:
        Tuple[bool, List[str]]: (검증 성공 여부, 에러 메시지 리스트)
    """
    errors = []

    for entry in migrations:
        source = Path(entry['source'])
        target_parent = Path(entry['target_parent'])
        target = Path(entry['target'])

        # 소스 존재 확인
        if not source.exists():
            errors.append(f"소스 디렉토리 없음: {source}")
            continue

        if not source.is_dir():
            errors.append(f"소스가 디렉토리가 아님: {source}")
            continue

        # 타겟 부모 디렉토리 확인
        if not target_parent.exists():
            errors.append(f"타겟 부모 디렉토리 없음: {target_parent}")
            continue

        if not target_parent.is_dir():
            errors.append(f"타겟 부모가 디렉토리가 아님: {target_parent}")
            continue

        # 타겟 디렉토리 중복 확인 (덮어쓰기 방지)
        if target.exists():
            errors.append(f"타겟 디렉토리 이미 존재 (덮어쓰기 방지): {target}")
            continue

        # .npy 파일 개수 카운트
        npy_files = list(source.glob('*.npy'))
        if len(npy_files) == 0:
            errors.append(f"소스에 .npy 파일 없음: {source}")
            continue

        entry['file_count'] = len(npy_files)
        entry['size_bytes'] = get_directory_size(str(source))

    return len(errors) == 0, errors


# =============================================================================
# 마이그레이션 함수
# =============================================================================

def migrate_single_directory(
    entry: Dict,
    dry_run: bool = True
) -> Dict:
    """
    단일 SMPL 디렉토리 이동

    Args:
        entry: 마이그레이션 정보 딕셔너리
        dry_run: True면 시뮬레이션만 수행

    Returns:
        Dict: 마이그레이션 결과 (entry 업데이트)
    """
    source = Path(entry['source'])
    target_parent = Path(entry['target_parent'])
    target = Path(entry['target'])

    result = entry.copy()
    result['timestamp'] = datetime.now().isoformat()

    if dry_run:
        result['status'] = 'dry_run'
        result['error'] = None
        return result

    try:
        # 디렉토리 이동
        shutil.move(str(source), str(target_parent))

        # 타겟 디렉토리 생성 확인
        if not target.exists():
            raise Exception("타겟 디렉토리 생성 실패")

        # 소스 삭제 확인
        if source.exists():
            raise Exception("소스 디렉토리 여전히 존재")

        # 타겟의 파일 개수 검증
        npy_files = list(target.glob('*.npy'))
        result['file_count_after'] = len(npy_files)

        if result.get('file_count') and result['file_count'] != result['file_count_after']:
            raise Exception(
                f"파일 개수 불일치: {result['file_count']} != {result['file_count_after']}"
            )

        result['status'] = 'success'
        result['error'] = None

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    return result


def execute_migration(migrations: List[Dict], dry_run: bool = True) -> List[Dict]:
    """
    전체 마이그레이션 프로세스 실행

    Args:
        migrations: 마이그레이션 정보 리스트
        dry_run: True면 시뮬레이션만 수행

    Returns:
        List[Dict]: 각 마이그레이션의 결과
    """
    results = []

    desc = "마이그레이션 실행 (시뮬레이션)" if dry_run else "마이그레이션 실행"

    pbar = tqdm(migrations, desc=desc, disable=dry_run and len(migrations) == 0)

    for entry in pbar:
        result = migrate_single_directory(entry, dry_run=dry_run)
        results.append(result)
        pbar.set_postfix(
            status=result['status'],
            success=sum(1 for r in results if r['status'] == 'success')
        )

    return results


# =============================================================================
# 매니페스트 함수
# =============================================================================

def create_manifest(
    migrations: List[Dict],
    mode: str,
    results: Optional[List[Dict]] = None
) -> Dict:
    """
    마이그레이션 매니페스트 생성

    Args:
        migrations: 마이그레이션 정보 리스트
        mode: 'dry_run' 또는 'execute'
        results: 실행 결과 (execute 모드에서만)

    Returns:
        Dict: 매니페스트 데이터
    """
    if results is None:
        results = migrations

    # 통계 계산
    total_size = sum(m.get('size_bytes', 0) for m in migrations)
    total_files = sum(m.get('file_count', 0) for m in migrations)

    successful = sum(1 for r in results if r.get('status') == 'success')
    failed = sum(1 for r in results if r.get('status') == 'failed')

    manifest = {
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'total_directories': len(migrations),
        'total_size_bytes': total_size,
        'total_files': total_files,
        'migrations': results,
        'summary': {
            'successful': successful,
            'failed': failed,
            'dry_run': mode == 'dry_run'
        }
    }

    return manifest


def save_manifest(manifest: Dict, output_dir: str) -> str:
    """
    매니페스트를 파일로 저장

    Args:
        manifest: 매니페스트 데이터
        output_dir: 출력 디렉토리

    Returns:
        str: 저장된 파일 경로
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"smpl_migration_manifest_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return filepath


# =============================================================================
# 검증 함수
# =============================================================================

def verify_migration(
    source_root: str,
    target_root: str,
    datasets: List[str],
    expected_count: int
) -> Tuple[bool, Dict]:
    """
    마이그레이션 후 검증

    Args:
        source_root: 소스 루트 디렉토리
        target_root: 타겟 루트 디렉토리
        datasets: 처리한 데이터셋 리스트
        expected_count: 예상 마이그레이션 디렉토리 개수

    Returns:
        Tuple[bool, Dict]: (검증 성공 여부, 검증 결과)
    """
    report = {
        'source_remaining': 0,
        'target_created': 0,
        'file_count_match': True,
        'errors': []
    }

    # 소스에 남은 smpl 디렉토리 확인
    for dataset in datasets:
        dataset_path = Path(source_root) / dataset
        if dataset_path.exists():
            for session_dir in dataset_path.iterdir():
                if (session_dir / 'processed_data' / 'smpl').exists():
                    report['source_remaining'] += 1
                    report['errors'].append(f"소스에 남은 smpl: {session_dir / 'processed_data' / 'smpl'}")

    # 타겟에 생성된 smpl 디렉토리 확인
    for dataset in datasets:
        dataset_path = Path(target_root) / dataset
        if dataset_path.exists():
            for session_dir in dataset_path.iterdir():
                if (session_dir / 'processed_data' / 'smpl').exists():
                    report['target_created'] += 1

    is_valid = (
        report['source_remaining'] == 0 and
        report['target_created'] == expected_count
    )

    return is_valid, report


# =============================================================================
# 출력 함수
# =============================================================================

def print_header(title: str):
    """헤더 출력"""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def print_section(title: str):
    """섹션 제목 출력"""
    print(f"\n--- {title} ---")


def print_statistics(migrations: List[Dict], results: Optional[List[Dict]] = None):
    """통계 출력"""
    if results is None:
        results = migrations

    # 데이터셋별 집계
    dataset_stats = {}
    for m in migrations:
        dataset = m['dataset']
        if dataset not in dataset_stats:
            dataset_stats[dataset] = {'count': 0, 'size': 0, 'files': 0}
        dataset_stats[dataset]['count'] += 1
        dataset_stats[dataset]['size'] += m.get('size_bytes', 0)
        dataset_stats[dataset]['files'] += m.get('file_count', 0)

    # 데이터셋별 통계 출력
    print_section("데이터셋별 통계")
    for dataset in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset]
        print(f"  {dataset}:")
        print(f"    - 디렉토리: {stats['count']}")
        print(f"    - 파일: {stats['files']}")
        print(f"    - 크기: {format_bytes(stats['size'])}")

    # 전체 통계
    print_section("전체 통계")
    total_size = sum(m.get('size_bytes', 0) for m in migrations)
    total_files = sum(m.get('file_count', 0) for m in migrations)

    print(f"  총 디렉토리: {len(migrations)}")
    print(f"  총 파일: {total_files}")
    print(f"  총 크기: {format_bytes(total_size)}")

    if results and results != migrations:
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        print(f"  성공: {successful}")
        print(f"  실패: {failed}")


# =============================================================================
# 메인 함수
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SMPL 디렉토리 마이그레이션 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # Dry-run 모드 (안전한 미리보기)
  python migrate_smpl_directories.py

  # 사전 검증만 실행
  python migrate_smpl_directories.py --verify-only

  # 실제 이동 실행
  python migrate_smpl_directories.py --execute

  # 커스텀 경로
  python migrate_smpl_directories.py \\
      --source-root /workspace/dataset \\
      --target-root /workspace/MMA/dataset \\
      --execute

  # 특정 데이터셋만 처리
  python migrate_smpl_directories.py \\
      --datasets 03_grappling2 \\
      --execute
        """
    )

    parser.add_argument(
        '--source-root',
        default=DEFAULT_SOURCE_ROOT,
        help=f'소스 루트 디렉토리 (기본값: {DEFAULT_SOURCE_ROOT})'
    )
    parser.add_argument(
        '--target-root',
        default=DEFAULT_TARGET_ROOT,
        help=f'타겟 루트 디렉토리 (기본값: {DEFAULT_TARGET_ROOT})'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=DEFAULT_DATASETS,
        help=f'처리할 데이터셋 (기본값: {" ".join(DEFAULT_DATASETS)})'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='실제 이동 실행 (없으면 dry-run)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='사전 검증만 실행 (마이그레이션 없음)'
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.dirname(__file__),
        help='매니페스트 파일 출력 디렉토리 (기본값: 스크립트 디렉토리)'
    )

    args = parser.parse_args()

    # 모드 결정
    if args.verify_only:
        mode = 'verify_only'
    elif args.execute:
        mode = 'execute'
    else:
        mode = 'dry_run'

    # 시작
    print_header("SMPL 디렉토리 마이그레이션 도구")
    print(f"모드: {mode.upper()}")
    print(f"소스: {args.source_root}")
    print(f"타겟: {args.target_root}")
    print(f"데이터셋: {', '.join(args.datasets)}")

    # Phase 1: Discovery
    print_section("Phase 1: 디렉토리 검색")
    migrations = discover_smpl_directories(args.source_root, args.target_root, args.datasets)
    print(f"발견된 SMPL 디렉토리: {len(migrations)}개")

    if not migrations:
        print("마이그레이션할 디렉토리가 없습니다.")
        return

    # 데이터셋별 분류
    dataset_counts = {}
    for m in migrations:
        dataset = m['dataset']
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    for dataset in sorted(dataset_counts.keys()):
        print(f"  {dataset}: {dataset_counts[dataset]}개")

    # Phase 2: Pre-flight Validation
    print_section("Phase 2: 사전 검증")
    is_valid, errors = verify_preconditions(migrations)

    if not is_valid:
        print("❌ 검증 실패:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("✓ 모든 소스 디렉토리 존재")
    print("✓ 모든 타겟 부모 디렉토리 존재")
    print("✓ 타겟에 기존 smpl 디렉토리 없음")

    # 통계 출력
    print_statistics(migrations)

    # verify-only 모드
    if mode == 'verify_only':
        print_header("검증 완료")
        print("✓ 모든 사전 조건 만족")
        return

    # Phase 3: Manifest Creation
    print_section("Phase 3: 매니페스트 생성")
    manifest = create_manifest(migrations, mode)
    manifest_path = save_manifest(manifest, args.output_dir)
    print(f"매니페스트 저장: {manifest_path}")

    # Dry-run 모드에서는 여기까지
    if mode == 'dry_run':
        print_header("요약")
        print("ℹ️  Dry-run 모드입니다. 파일이 변경되지 않았습니다.")
        print(f"실제 마이그레이션을 실행하려면 --execute 플래그를 사용하세요.")
        return

    # Phase 4: Migration Execution
    print_section("Phase 4: 마이그레이션 실행")
    results = execute_migration(migrations, dry_run=False)

    # Phase 5: Post-migration Verification
    print_section("Phase 5: 마이그레이션 후 검증")
    is_verified, verification_report = verify_migration(
        args.source_root,
        args.target_root,
        args.datasets,
        len(migrations)
    )

    print(f"소스에 남은 smpl 디렉토리: {verification_report['source_remaining']}")
    print(f"타겟에 생성된 smpl 디렉토리: {verification_report['target_created']}")

    if verification_report['errors']:
        print("⚠️  경고:")
        for error in verification_report['errors']:
            print(f"  - {error}")

    # 최종 결과
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')

    print_header("마이그레이션 결과")
    print(f"✓ 성공: {successful}/{len(migrations)}")
    if failed > 0:
        print(f"❌ 실패: {failed}/{len(migrations)}")

    print_statistics(migrations, results)

    print_section("매니페스트")
    print(f"파일: {manifest_path}")

    if is_verified:
        print_header("모든 검증 통과 ✓")
    else:
        print_header("검증 경고 ⚠️")


if __name__ == '__main__':
    main()
