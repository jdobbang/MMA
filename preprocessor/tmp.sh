#!/bin/bash

# 1. 대상 디렉토리 설정
TARGET_BASE="/workspace/dataset"
TEMP_DIR="./temp_extract"

# 대상 디렉토리 생성
mkdir -p "$TARGET_BASE"
mkdir -p "$TEMP_DIR"

# 2. 다운로드할 파일 리스트 (Hugging Face resolve 링크로 변환)
URLS=(
    "https://huggingface.co/datasets/Jyun-Ting/Harmony4D/resolve/main/train/13_mma2.zip"
    "https://huggingface.co/datasets/Jyun-Ting/Harmony4D/resolve/main/train/03_grappling2.zip"
)

for URL in "${URLS[@]}"; do
    FILE_NAME=$(basename "$URL")
    DIR_NAME="${FILE_NAME%.zip}" # 13_mma2, 03_grappling2 등
    
    echo "-------------------------------------------------------"
    echo "시작: $FILE_NAME 다운로드 중..."
    echo "-------------------------------------------------------"
    
    # 3. 파일 다운로드 (wget 설치 필요)
    wget -c "$URL" -O "$FILE_NAME"
    
    echo "압축 해제 중: $FILE_NAME..."
    # 4. 임시 폴더에 압축 해제 (기존 내용 삭제 후 진행)
    rm -rf "$TEMP_DIR"/*
    unzip -q "$FILE_NAME" -d "$TEMP_DIR"
    
    echo "데이터 필터링 및 이동 중 (colmap 폴더만 추출)..."
    
    # 5. colmap 폴더만 찾아 상위 구조 유지하며 이동
    # find로 'colmap' 폴더를 찾고, 해당 경로를 보존하며 복사합니다.
    cd "$TEMP_DIR"
    
    # 13_mma2와 같은 루트 폴더 내부로 진입하여 작업
    if [ -d "$DIR_NAME" ]; then
        cd "$DIR_NAME"
        
        # colmap 폴더가 있는 경로들을 찾아 복사
        # /workspace/dataset/13_mma2/하위폴더/processed_data/colmap 구조 생성
        find . -type d -path "*/processed_data/colmap" | while read -r colmap_path; do
            DEST_PATH="$TARGET_BASE/$DIR_NAME/${colmap_path#./}"
            mkdir -p "$(dirname "$DEST_PATH")"
            cp -r "$colmap_path" "$(dirname "$DEST_PATH")/"
        done
        
        cd ../..
    else
        echo "경고: 압축 해제 후 $DIR_NAME 폴더를 찾을 수 없습니다."
        cd ..
    fi
    
    # 6. 중간 단계 파일 삭제 (용량 확보)
    echo "임시 파일 정리 중..."
    rm -f "$FILE_NAME"
    rm -rf "$TEMP_DIR"/*
    
    echo "완료: $DIR_NAME 작업이 끝났습니다."
done

echo "-------------------------------------------------------"
echo "모든 작업이 완료되었습니다!"
echo "위치: $TARGET_BASE"
echo "-------------------------------------------------------"