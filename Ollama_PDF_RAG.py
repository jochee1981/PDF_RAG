# Ollama_PDF_RAG.py
import gradio as gr
import ollama
import os
import hashlib
import shutil
import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# 설정 파일 로드
def load_config():
    """설정 파일 로드"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # 기본 설정 반환
        return {
            'ollama': {'llm_model': 'llama3', 'embedding_model': 'mxbai-embed-large'},
            'text_splitter': {'chunk_size': 1000, 'chunk_overlap': 200},
            'vectorstore': {'persist_directory': 'chroma_pdf_cache', 'search_kwargs': {'k': 4}},
            'pdf': {'storage_directory': 'pdfs'},
            'rag': {'enable_cache': True, 'system_prompt': 'You are a helpful assistant. Read the PDF content and answer the question. Translate the answer in Korean with emoji.'},
            'logging': {'level': 'INFO'}
        }

config = load_config()

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, config.get('logging', {}).get('level', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 디렉토리 설정
PDF_STORAGE_DIR = config.get('pdf', {}).get('storage_directory', 'pdfs')
VECTOR_CACHE_DIR = config.get('vectorstore', {}).get('persist_directory', 'chroma_pdf_cache')

# 디렉토리 생성
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)
os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)

def sanitize_filename(filename: str) -> str:
    """파일명을 안전하게 변환 (폴더명으로 사용 가능하도록)"""
    # 확장자 분리
    name, ext = os.path.splitext(filename)
    
    # 특수문자 제거 및 공백을 언더스코어로 변환
    # Windows에서 사용할 수 없는 문자: < > : " / \ | ? *
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)  # 연속된 언더스코어를 하나로
    name = name.strip('_')  # 앞뒤 언더스코어 제거
    
    # 빈 이름 처리
    if not name:
        name = "unnamed"
    
    # 길이 제한 (Windows 경로 길이 제한 고려)
    if len(name) > 200:
        name = name[:200]
    
    return f"{name}{ext}"

def get_file_hash(file_path: str) -> str:
    """파일 해시 계산"""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def save_metadata(pdf_folder: str, metadata: dict):
    """메타데이터를 JSON 파일로 저장"""
    metadata_path = os.path.join(pdf_folder, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.debug(f"메타데이터 저장 완료: {metadata_path}")

def load_metadata(pdf_folder: str) -> Optional[dict]:
    """메타데이터 파일 로드"""
    metadata_path = os.path.join(pdf_folder, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"메타데이터 로드 실패: {str(e)}")
    return None

def copy_pdf_to_storage(file_path: str) -> Tuple[str, str, str]:
    """PDF 파일을 저장 디렉토리로 복사 (폴더별로 저장)
    
    Returns:
        Tuple[stored_path, pdf_folder, file_hash]
    """
    original_filename = os.path.basename(file_path)
    safe_filename = sanitize_filename(original_filename)
    
    # 파일 해시 계산 (원본 파일 기준)
    file_hash = get_file_hash(file_path)
    
    # 폴더명 생성 (파일명 기반)
    name_without_ext = os.path.splitext(safe_filename)[0]
    pdf_folder = os.path.join(PDF_STORAGE_DIR, name_without_ext)
    
    # 폴더가 이미 존재하는 경우, 해시로 확인
    if os.path.exists(pdf_folder):
        existing_metadata = load_metadata(pdf_folder)
        if existing_metadata and existing_metadata.get('file_hash') == file_hash:
            # 동일한 파일이면 기존 폴더 사용
            stored_path = os.path.join(pdf_folder, 'original.pdf')
            logger.info(f"기존 PDF 폴더 사용: {pdf_folder}")
            return stored_path, pdf_folder, file_hash
        else:
            # 다른 파일이면 새 폴더명 생성
            counter = 1
            while os.path.exists(pdf_folder):
                new_folder_name = f"{name_without_ext}_{counter}"
                pdf_folder = os.path.join(PDF_STORAGE_DIR, new_folder_name)
                counter += 1
            logger.info(f"새 PDF 폴더 생성: {pdf_folder}")
    
    # 폴더 생성
    os.makedirs(pdf_folder, exist_ok=True)
    
    # PDF 파일 복사
    stored_path = os.path.join(pdf_folder, 'original.pdf')
    shutil.copy2(file_path, stored_path)
    
    # 메타데이터 저장
    file_size = os.path.getsize(stored_path)
    metadata = {
        'original_filename': original_filename,
        'safe_filename': safe_filename,
        'file_hash': file_hash,
        'file_size': file_size,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    save_metadata(pdf_folder, metadata)
    
    logger.info(f"PDF 파일 복사 완료: {stored_path} (폴더: {pdf_folder})")
    return stored_path, pdf_folder, file_hash

def check_ollama_connection() -> Tuple[bool, str]:
    """Ollama 연결 확인"""
    try:
        ollama.list()
        return True, "Ollama 연결 성공"
    except Exception as e:
        return False, f"Ollama 연결 실패: {str(e)}"

def check_model_exists(model_name: str) -> Tuple[bool, str]:
    """모델 존재 여부 확인"""
    try:
        models_response = ollama.list()
        # ollama.list()는 {'models': [...]} 형식 또는 리스트를 반환할 수 있음
        if isinstance(models_response, dict):
            models = models_response.get('models', [])
        else:
            models = models_response
        
        model_names = []
        for model in models:
            if isinstance(model, dict):
                model_names.append(model.get('name', ''))
            else:
                model_names.append(str(model))
        
        if model_name in model_names:
            return True, f"모델 '{model_name}' 확인됨"
        else:
            available = ', '.join(model_names) if model_names else "없음"
            return False, f"모델 '{model_name}'을 찾을 수 없습니다. 설치된 모델: {available}"
    except Exception as e:
        return False, f"모델 확인 실패: {str(e)}"

def load_and_retrieve_pdf(file_path: str, use_cache: bool = True, pdf_folder: Optional[str] = None):
    """PDF 문서 로드 및 벡터화 (캐시 지원, 폴더별 관리)
    
    Args:
        file_path: PDF 파일 경로
        use_cache: 캐시 사용 여부
        pdf_folder: 이미 저장된 PDF 폴더 경로 (선택사항)
    """
    # PDF를 저장 디렉토리로 복사 (아직 저장되지 않은 경우만)
    if pdf_folder is None or not os.path.exists(pdf_folder):
        stored_path, pdf_folder, file_hash = copy_pdf_to_storage(file_path)
    else:
        stored_path = os.path.join(pdf_folder, 'original.pdf')
        file_hash = get_file_hash(stored_path)
    
    # 파일명 기반 벡터스토어 폴더명 생성
    folder_name = os.path.basename(pdf_folder)
    vectorstore_dir = os.path.join(VECTOR_CACHE_DIR, f"{folder_name}_{file_hash[:8]}")
    
    # 메타데이터 로드
    metadata = load_metadata(pdf_folder)
    
    # 캐시 확인
    if use_cache and os.path.exists(vectorstore_dir) and os.path.exists(os.path.join(vectorstore_dir, 'chroma.sqlite3')):
        logger.info(f"캐시된 벡터스토어 사용: {vectorstore_dir}")
        try:
            embeddings = OllamaEmbeddings(model=config['ollama']['embedding_model'])
            vectorstore = Chroma(
                persist_directory=vectorstore_dir,
                embedding_function=embeddings
            )
            
            # 메타데이터 업데이트 (마지막 사용 시간)
            if metadata:
                metadata['last_accessed_at'] = datetime.now().isoformat()
                save_metadata(pdf_folder, metadata)
            
            return vectorstore.as_retriever(search_kwargs=config.get('vectorstore', {}).get('search_kwargs', {'k': 4}))
        except Exception as e:
            logger.warning(f"캐시 로드 실패, 재벡터화 진행: {str(e)}")
    
    # PDF 로드
    logger.info(f"PDF 문서 로드 시작: {stored_path}")
    loader = PyMuPDFLoader(stored_path)
    docs = loader.load()
    
    if not docs:
        raise ValueError("❗ PDF에서 텍스트를 추출할 수 없습니다. 다른 파일을 시도해 보세요.")
    
    logger.info(f"PDF 문서 로드 완료. 총 {len(docs)} 페이지")
    logger.debug(f"첫 페이지 미리보기:\n{docs[0].page_content[:300]}...")
    
    # 텍스트 분할
    chunk_size = config.get('text_splitter', {}).get('chunk_size', 1000)
    chunk_overlap = config.get('text_splitter', {}).get('chunk_overlap', 200)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    logger.info(f"텍스트 분할 완료. 총 {len(splits)}개 청크")
    
    # 임베딩 및 벡터스토어 생성
    embeddings = OllamaEmbeddings(model=config['ollama']['embedding_model'])
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=vectorstore_dir
    )
    vectorstore.persist()
    logger.info(f"벡터스토어 생성 완료: {vectorstore_dir}")
    
    # 메타데이터 업데이트
    if metadata:
        metadata['page_count'] = len(docs)
        metadata['chunk_count'] = len(splits)
        metadata['vectorized_at'] = datetime.now().isoformat()
        metadata['last_accessed_at'] = datetime.now().isoformat()
        metadata['vectorstore_path'] = vectorstore_dir
    else:
        metadata = {
            'page_count': len(docs),
            'chunk_count': len(splits),
            'vectorized_at': datetime.now().isoformat(),
            'last_accessed_at': datetime.now().isoformat(),
            'vectorstore_path': vectorstore_dir
        }
    save_metadata(pdf_folder, metadata)
    
    return vectorstore.as_retriever(search_kwargs=config.get('vectorstore', {}).get('search_kwargs', {'k': 4}))

def format_docs(docs):
    """문서 포맷팅"""
    return "\n\n".join(f"[문서 {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs))

def rag_chain(files: List[gr.File], question: str) -> str:
    """RAG 체인 동작 (다중 PDF 지원, 폴더별 관리)"""
    try:
        # Ollama 연결 확인
        connected, msg = check_ollama_connection()
        if not connected:
            return f"❌ {msg}\n\nOllama 서비스가 실행 중인지 확인해주세요."
        
        # 모델 확인
        llm_model = config['ollama']['llm_model']
        model_exists, model_msg = check_model_exists(llm_model)
        if not model_exists:
            return f"❌ {model_msg}\n\n모델을 설치하려면: ollama pull {llm_model}"
        
        embedding_model = config['ollama']['embedding_model']
        embedding_exists, embedding_msg = check_model_exists(embedding_model)
        if not embedding_exists:
            return f"❌ {embedding_msg}\n\n모델을 설치하려면: ollama pull {embedding_model}"
        
        if not files or len(files) == 0:
            return "❌ PDF 파일을 선택해주세요."
        
        if not question or question.strip() == "":
            return "❌ 질문을 입력해주세요."
        
        # 다중 PDF 처리
        all_retrieved_docs = []
        processed_files = []
        processed_folders = []
        
        for file in files:
            if file is None:
                continue
                
            file_path = file.name if hasattr(file, 'name') else file
            if not os.path.exists(file_path):
                logger.warning(f"파일을 찾을 수 없음: {file_path}")
                continue
            
            try:
                logger.info(f"PDF 처리 시작: {os.path.basename(file_path)}")
                # PDF를 저장하고 폴더 정보 가져오기
                stored_path, pdf_folder, _ = copy_pdf_to_storage(file_path)
                folder_name = os.path.basename(pdf_folder)
                
                # 벡터스토어 로드 및 검색 (이미 저장된 폴더 정보 전달)
                retriever = load_and_retrieve_pdf(file_path, use_cache=config.get('rag', {}).get('enable_cache', True), pdf_folder=pdf_folder)
                retrieved_docs = retriever.invoke(question)
                
                if retrieved_docs:
                    all_retrieved_docs.extend(retrieved_docs)
                    filename = os.path.basename(file_path)
                    processed_files.append(filename)
                    processed_folders.append(folder_name)
                    
                    logger.info(f"'{filename}'에서 {len(retrieved_docs)}개 문서 검색됨 (폴더: {folder_name})")
            except Exception as e:
                logger.error(f"파일 처리 중 오류 발생 ({os.path.basename(file_path)}): {str(e)}")
                continue
        
        if not all_retrieved_docs:
            return "❌ 관련 문서를 찾을 수 없습니다. 질문을 더 구체적으로 작성해 보거나 다른 PDF를 사용해 보세요."
        
        # 중복 제거 (동일한 내용의 문서)
        seen = set()
        unique_docs = []
        for doc in all_retrieved_docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)
        
        logger.info(f"총 {len(unique_docs)}개 고유 문서 검색됨 (처리된 파일: {', '.join(processed_files)})")
        
        # 컨텍스트 구성
        context = format_docs(unique_docs)
        logger.debug(f"검색된 문맥 미리보기:\n{context[:500]}...")
        
        # 프롬프트 생성
        file_list = "\n".join([f"- {f}" for f in processed_files])
        prompt = f"""다음 PDF 파일들에서 검색된 내용을 바탕으로 질문에 답변해주세요.

처리된 PDF 파일:
{file_list}

질문: {question}

검색된 내용:
{context}"""
        
        # LLM 호출
        logger.info(f"LLM 호출 시작 (모델: {llm_model})")
        response = ollama.chat(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": config.get('rag', {}).get('system_prompt', 'You are a helpful assistant.')
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        result = response['message']['content']
        logger.info("LLM 응답 완료")
        
        # 처리된 파일 정보 추가
        folder_info = f"📁 저장 폴더: {', '.join(processed_folders)}" if processed_folders else ""
        result_with_info = f"📄 처리된 파일: {', '.join(processed_files)}\n{folder_info}\n\n{result}"
        
        return result_with_info
        
    except Exception as e:
        error_msg = f"❌ 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

# Gradio 인터페이스
def create_interface():
    """Gradio 인터페이스 생성"""
    with gr.Blocks(title="PDF RAG 시스템") as iface:
        gr.Markdown("# 📚 다중 PDF 기반 질문 응답 시스템")
        gr.Markdown("여러 PDF 파일을 선택하고 질문을 입력하면, 해당 내용을 기반으로 답변해 드립니다.")
        gr.Markdown("**각 PDF는 별도 폴더로 관리되며, 파일명과 해시를 기반으로 캐시됩니다.**")
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="PDF 파일 선택 (여러 파일 선택 가능)",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath"
                )
                question_input = gr.Textbox(
                    label="질문을 입력하세요",
                    placeholder="예: 이 문서의 주요 내용은 무엇인가요?",
                    lines=3
                )
                submit_btn = gr.Button("질문하기", variant="primary")
            
            with gr.Column():
                output = gr.Textbox(
                    label="답변",
                    lines=15,
                    interactive=False
                )
        
        gr.Markdown("### 💡 사용 팁")
        gr.Markdown("""
        - 여러 PDF 파일을 동시에 선택할 수 있습니다
        - 각 PDF는 `pdfs/{파일명}/` 폴더에 저장됩니다
        - 동일한 파일은 캐시를 사용하여 빠르게 처리됩니다
        - 각 PDF 폴더에는 메타데이터가 저장됩니다
        - 질문을 구체적으로 작성할수록 더 정확한 답변을 받을 수 있습니다
        """)
        
        submit_btn.click(
            fn=rag_chain,
            inputs=[file_input, question_input],
            outputs=output
        )
        
        # Enter 키로도 제출 가능
        question_input.submit(
            fn=rag_chain,
            inputs=[file_input, question_input],
            outputs=output
        )
    
    return iface

if __name__ == "__main__":
    # 시작 시 Ollama 연결 확인
    connected, msg = check_ollama_connection()
    if not connected:
        logger.warning(msg)
        print(f"⚠️ 경고: {msg}")
    
    iface = create_interface()
    iface.launch(share=False)
