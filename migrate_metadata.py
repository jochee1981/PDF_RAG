# migrate_metadata.py - 기존 PDF 파일들에 메타데이터 생성
import os
import json
import hashlib
from datetime import datetime

def get_file_hash(file_path: str) -> str:
    """파일 해시 계산"""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def migrate_pdfs():
    """기존 PDF 파일들에 메타데이터 생성"""
    pdfs_dir = 'pdfs'
    
    if not os.path.exists(pdfs_dir):
        print(f"'{pdfs_dir}' 디렉토리가 존재하지 않습니다.")
        return
    
    migrated_count = 0
    for item in os.listdir(pdfs_dir):
        item_path = os.path.join(pdfs_dir, item)
        
        # 폴더인 경우만 처리
        if os.path.isdir(item_path):
            original_pdf = os.path.join(item_path, 'original.pdf')
            metadata_path = os.path.join(item_path, 'metadata.json')
            
            # original.pdf가 존재하는 경우
            if os.path.exists(original_pdf):
                # 메타데이터가 없으면 생성
                if not os.path.exists(metadata_path):
                    file_hash = get_file_hash(original_pdf)
                    file_size = os.path.getsize(original_pdf)
                    
                    metadata = {
                        'original_filename': item,  # 폴더명이 원래 파일명의 변환된 버전
                        'safe_filename': f"{item}.pdf",
                        'file_hash': file_hash,
                        'file_size': file_size,
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat(),
                        'migrated': True
                    }
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                    print(f"✓ 메타데이터 생성: {item_path}")
                    migrated_count += 1
                else:
                    print(f"- 메타데이터 이미 존재: {item_path}")
            else:
                print(f"⚠ original.pdf를 찾을 수 없음: {item_path}")
    
    print(f"\n총 {migrated_count}개 폴더에 메타데이터를 생성했습니다.")

if __name__ == "__main__":
    migrate_pdfs()

