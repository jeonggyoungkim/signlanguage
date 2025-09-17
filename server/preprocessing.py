import json
import numpy as np
import os
import glob
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.interpolate import interp1d
from functools import lru_cache
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedKeypointProcessor:
    """최적화된 키포인트 처리기"""
    
    # 클래스 상수 (메모리 효율성)
    POSE_KEYPOINTS = 25
    FACE_KEYPOINTS = 70
    HAND_KEYPOINTS = 21
    TOTAL_KEYPOINTS = POSE_KEYPOINTS + FACE_KEYPOINTS + (HAND_KEYPOINTS * 2)
    
    # 키포인트 매핑 (딕셔너리 대신 튜플 사용으로 메모리 절약)
    KEYPOINT_PARTS = ('pose_keypoints_2d', 'face_keypoints_2d', 
                     'hand_left_keypoints_2d', 'hand_right_keypoints_2d')
    KEYPOINT_COUNTS = (POSE_KEYPOINTS, FACE_KEYPOINTS, HAND_KEYPOINTS, HAND_KEYPOINTS)
    
    def __init__(self, image_width: int = 1920, image_height: int = 1080, 
                 target_frames: int = 180, target_batch_size: int = 15,
                 enable_multiprocessing: bool = True, max_workers: int = None):
        """
        최적화된 키포인트 처리기 초기화
        
        Args:
            image_width: 원본 이미지 너비
            image_height: 원본 이미지 높이
            target_frames: 목표 프레임 수
            target_batch_size: 목표 배치 사이즈
            enable_multiprocessing: 멀티프로세싱 활성화 여부
            max_workers: 최대 워커 수 (None이면 CPU 코어 수)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.target_frames = target_frames
        self.target_batch_size = target_batch_size
        self.enable_multiprocessing = enable_multiprocessing
        self.max_workers = max_workers or min(8, os.cpu_count() or 1)
        
        # 정규화 상수 미리 계산 (성능 최적화)
        self.inv_width = 1.0 / image_width
        self.inv_height = 1.0 / image_height
        
        logger.info(f"키포인트 구성: Pose({self.POSE_KEYPOINTS}) + Face({self.FACE_KEYPOINTS}) + "
                   f"Hands({self.HAND_KEYPOINTS}×2) = {self.TOTAL_KEYPOINTS}")
        logger.info(f"멀티프로세싱: {'활성화' if enable_multiprocessing else '비활성화'} "
                   f"(최대 워커: {self.max_workers})")
    
    @lru_cache(maxsize=128)
    def _load_json_cached(self, json_path: str) -> str:
        """JSON 파일 캐싱 로드 (작은 파일들의 중복 로드 방지)"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_keypoints(self, json_path: str) -> Dict:
        """JSON 파일에서 키포인트 데이터 로드 (캐싱 적용)"""
        try:
            json_str = self._load_json_cached(json_path)
            return json.loads(json_str)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"JSON 로드 실패: {json_path} - {e}")
            raise
    
    def extract_2d_keypoints_vectorized(self, data: Dict) -> np.ndarray:
        """
        벡터화된 2D 키포인트 추출 (성능 최적화)
        
        Returns:
            shape: (total_keypoints, 3) - 모든 키포인트를 하나의 배열로
        """
        people_data = data['people']
        
        # 모든 키포인트를 한 번에 처리
        keypoints_list = []
        for part_name, count in zip(self.KEYPOINT_PARTS, self.KEYPOINT_COUNTS):
            keypoint_data = np.array(people_data[part_name], dtype=np.float32)
            keypoints_list.append(keypoint_data.reshape(count, 3))
        
        # 한 번에 concatenate (메모리 효율적)
        return np.vstack(keypoints_list)
    
    def normalize_keypoints_vectorized(self, keypoints: np.ndarray) -> np.ndarray:
        """
        벡터화된 키포인트 정규화 (성능 최적화)
        
        Args:
            keypoints: (N, 3) 형태의 키포인트 배열
            
        Returns:
            정규화된 키포인트 배열
        """
        # in-place 연산으로 메모리 절약
        normalized = keypoints.copy()
        normalized[:, 0] *= self.inv_width   # x 정규화
        normalized[:, 1] *= self.inv_height  # y 정규화
        return normalized
    
    def convert_to_relative_vectorized(self, keypoints: np.ndarray) -> np.ndarray:
        """
        벡터화된 상대 위치 변환 (성능 최적화)
        
        Args:
            keypoints: (total_keypoints, 3) 정규화된 키포인트
            
        Returns:
            코 기준 상대 위치 키포인트
        """
        # 코(nose) 위치 찾기 (첫 번째 키포인트)
        nose_keypoint = keypoints[0]
        
        if nose_keypoint[2] <= 0:  # confidence 체크
            logger.warning("코(nose) 키포인트의 confidence가 0입니다.")
            return keypoints
        
        # 벡터화된 상대 위치 계산
        relative_keypoints = keypoints.copy()
        relative_keypoints[:, :2] -= nose_keypoint[:2]  # x, y 좌표만 변환
        
        return relative_keypoints
    
    def process_single_frame_optimized(self, json_path: str) -> np.ndarray:
        """
        최적화된 단일 프레임 처리
        
        Returns:
            shape: (total_keypoints, 2) - x, y 좌표만
        """
        # 1. 키포인트 로드 및 벡터화된 추출
        data = self.load_keypoints(json_path)
        keypoints = self.extract_2d_keypoints_vectorized(data)
        
        # 2. 벡터화된 정규화 및 상대위치 변환
        normalized = self.normalize_keypoints_vectorized(keypoints)
        relative = self.convert_to_relative_vectorized(normalized)
        
        # 3. x, y 좌표만 반환
        return relative[:, :2]
    
    def resample_sequence_optimized(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """
        최적화된 시퀀스 리샘플링
        
        Args:
            keypoints_sequence: shape (original_frames, keypoints, 2)
            
        Returns:
            shape: (target_frames, keypoints, 2)
        """
        original_frames = keypoints_sequence.shape[0]
        
        if original_frames == self.target_frames:
            return keypoints_sequence
        
        # 벡터화된 보간 (모든 키포인트와 좌표를 한 번에 처리)
        original_indices = np.linspace(0, original_frames - 1, original_frames)
        target_indices = np.linspace(0, original_frames - 1, self.target_frames)
        
        # reshape로 2D 배열로 만들어 한 번에 보간
        flat_sequence = keypoints_sequence.reshape(original_frames, -1)
        
        # 모든 차원을 한 번에 보간
        interpolator = interp1d(original_indices, flat_sequence, 
                              kind='linear', axis=0, 
                              bounds_error=False, fill_value='extrapolate')
        
        resampled_flat = interpolator(target_indices)
        
        # 원래 형태로 복원
        return resampled_flat.reshape(self.target_frames, self.TOTAL_KEYPOINTS, 2)
    
    def process_json_files_batch(self, json_files: List[str]) -> List[np.ndarray]:
        """
        배치 JSON 파일 처리 (멀티프로세싱 지원)
        
        Args:
            json_files: JSON 파일 경로 리스트
            
        Returns:
            처리된 키포인트 리스트
        """
        if not self.enable_multiprocessing or len(json_files) < 4:
            # 단일 스레드 처리
            return [self.process_single_frame_optimized(json_file) 
                   for json_file in json_files]
        
        # 멀티스레드 처리
        keypoints_list = [None] * len(json_files)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 순서 보장을 위해 인덱스와 함께 처리
            future_to_index = {
                executor.submit(self.process_single_frame_optimized, json_files[i]): i 
                for i in range(len(json_files))
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    keypoints_list[index] = future.result()
                except Exception as e:
                    logger.error(f"프레임 처리 실패 (인덱스 {index}): {e}")
                    # 빈 키포인트로 대체
                    keypoints_list[index] = np.zeros((self.TOTAL_KEYPOINTS, 2), dtype=np.float32)
        
        return keypoints_list
    
    def find_json_files_sorted(self, root_dir: str) -> Dict[str, List[str]]:
        """
        최적화된 JSON 파일 찾기 및 정렬
        
        Returns:
            폴더명: 정렬된 JSON 파일 리스트 딕셔너리
        """
        folder_json_map = {}
        
        try:
            subdirs = [d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))]
        except OSError as e:
            logger.error(f"디렉토리 읽기 실패: {root_dir} - {e}")
            return {}
        
        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            json_pattern = os.path.join(subdir_path, "*_keypoints.json")
            json_files = glob.glob(json_pattern)
            
            if json_files:
                # 최적화된 정렬 (숫자 추출을 한 번만)
                def extract_frame_number(filepath):
                    try:
                        return int(os.path.basename(filepath).split('_')[-2])
                    except (ValueError, IndexError):
                        return 0
                
                json_files.sort(key=extract_frame_number)
                folder_json_map[subdir] = json_files
                logger.info(f"폴더 '{subdir}': {len(json_files)}개 JSON 파일")
        
        logger.info(f"총 {len(folder_json_map)}개 폴더에서 JSON 파일 발견")
        return folder_json_map
    
    def process_video_sequence_optimized(self, json_files: List[str]) -> np.ndarray:
        """
        최적화된 비디오 시퀀스 처리
        
        Returns:
            shape: (target_frames, keypoints*2) - 평탄화된 시퀀스
        """
        # 배치 처리로 모든 프레임을 한 번에 처리
        keypoints_list = self.process_json_files_batch(json_files)
        
        # 유효한 키포인트만 필터링
        valid_keypoints = [kp for kp in keypoints_list if kp is not None]
        
        if not valid_keypoints:
            raise ValueError("유효한 키포인트가 없습니다.")
        
        # numpy 배열로 변환
        keypoints_sequence = np.stack(valid_keypoints, axis=0)
        
        # 최적화된 리샘플링
        resampled = self.resample_sequence_optimized(keypoints_sequence)
        
        # 평탄화: (frames, keypoints, 2) -> (frames, keypoints*2)
        return resampled.reshape(self.target_frames, -1)
    
    def create_csv_optimized(self, tensor: np.ndarray, folder_name: str, 
                           output_dir: str) -> Dict[str, str]:
        """
        최적화된 CSV 생성 (메모리 효율적)
        
        Args:
            tensor: shape (target_frames, keypoints*2)
            folder_name: 폴더명
            output_dir: 출력 디렉토리
            
        Returns:
            생성된 파일 정보
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 컬럼명 미리 생성 (캐싱)
        if not hasattr(self, '_csv_columns'):
            self._csv_columns = ['frame'] + [
                f"keypoint_{i}_{coord}" 
                for i in range(self.TOTAL_KEYPOINTS) 
                for coord in ['x', 'y']
            ]
        
        # 메모리 효율적인 DataFrame 생성
        data_dict = {'frame': range(1, self.target_frames + 1)}
        
        # 컬럼별로 데이터 할당 (메모리 효율적)
        for i, col in enumerate(self._csv_columns[1:]):
            data_dict[col] = tensor[:, i]
        
        df = pd.DataFrame(data_dict)
        
        # CSV 저장
        csv_filename = f"{folder_name}_keypoints.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        # 메모리 효율적인 CSV 저장
        df.to_csv(csv_path, index=False, float_format='%.4f')
        
        logger.info(f"CSV 저장: {csv_filename} - {tensor.shape}")
        
        return {
            'csv_file': csv_filename,
            'csv_path': csv_path,
            'shape': tensor.shape
        }
    
    def process_single_video_optimized(self, video_folder_path: str, 
                                     output_dir: str = "optimized_output") -> Dict:
        """
        최적화된 단일 비디오 처리 (메인 함수)
        
        Args:
            video_folder_path: 비디오 폴더 경로
            output_dir: 출력 디렉토리
            
        Returns:
            처리 결과 정보
        """
        logger.info(f"=== 최적화된 단일 비디오 처리 시작 ===")
        logger.info(f"폴더: {video_folder_path}")
        
        # JSON 파일 찾기
        json_pattern = os.path.join(video_folder_path, "*_keypoints.json")
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            raise ValueError(f"JSON 파일을 찾을 수 없습니다: {video_folder_path}")
        
        # 프레임 번호로 정렬
        json_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-2]))
        
        folder_name = os.path.basename(video_folder_path)
        logger.info(f"처리할 프레임: {len(json_files)}개 -> {self.target_frames}개로 리샘플링")
        
        # 최적화된 비디오 시퀀스 처리
        tensor = self.process_video_sequence_optimized(json_files)
        
        # CSV 생성
        csv_info = self.create_csv_optimized(tensor, folder_name, output_dir)
        
        result = {
            **csv_info,
            'original_frames': len(json_files),
            'resampled_frames': self.target_frames,
            'folder_name': folder_name,
            'processing_time': 'optimized'
        }
        
        logger.info("=== 처리 완료 ===")
        return result

def process_video_optimized(video_folder_path: str, output_dir: str = "optimized_output",
                          **kwargs) -> Dict:
    """
    최적화된 비디오 처리 함수 (편의 함수)
    
    Args:
        video_folder_path: 비디오 폴더 경로
        output_dir: 출력 디렉토리
        **kwargs: OptimizedKeypointProcessor 초기화 인자
        
    Returns:
        처리 결과
    """
    processor = OptimizedKeypointProcessor(**kwargs)
    return processor.process_single_video_optimized(video_folder_path, output_dir)

def main():
    """최적화된 메인 함수"""
    # 설정
    # target_video_folder = "Source_data/9.슬프다/NIA_SL_WORD0009_REAL15_F" # 기존 단일 비디오 폴더
    word_folder_path = "Source_data/9.슬프다"  # 처리할 단어 폴더 경로로 변경
    output_dir = "optimized_output"  # CSV 파일이 저장될 기본 출력 디렉토리

    if not os.path.exists(word_folder_path):
        logger.error(f"단어 폴더를 찾을 수 없습니다: {word_folder_path}")
        return

    # 단어 폴더 내의 비디오 폴더(하위 디렉토리) 목록 가져오기
    try:
        video_folders = [
            os.path.join(word_folder_path, d)
            for d in os.listdir(word_folder_path)
            if os.path.isdir(os.path.join(word_folder_path, d))
        ]
    except OSError as e:
        logger.error(f"단어 폴더 '{word_folder_path}'의 내용을 읽는 중 오류 발생: {e}")
        return

    if not video_folders:
        logger.info(f"'{word_folder_path}' 내에 처리할 비디오 폴더가 없습니다.")
        return

    logger.info(f"'{word_folder_path}' 에서 다음 {len(video_folders)}개의 비디오 폴더를 처리합니다:")
    for vf in video_folders:
        logger.info(f"  - {os.path.basename(vf)}")

    all_results = []
    for video_folder_path_item in video_folders:
        logger.info(f"--- '{os.path.basename(video_folder_path_item)}' 처리 시작 ---")
        try:
            # 최적화된 처리 실행
            # process_video_optimized 함수는 내부적으로 OptimizedKeypointProcessor를 생성하며,
            # 이때 enable_multiprocessing 및 max_workers가 전달됩니다.
            result = process_video_optimized(
                video_folder_path=video_folder_path_item,
                output_dir=output_dir,  # 모든 CSV가 이 디렉토리에 저장됨
                enable_multiprocessing=True,
                max_workers=4  # 필요시 이 값을 조정하거나 인스턴스 생성시 CPU 코어 수 기반으로 자동 설정됨
            )
            
            if result: # result가 None이 아닐 경우 (오류 없이 처리된 경우)
                logger.info(f"--- '{os.path.basename(video_folder_path_item)}' 처리 완료 ---")
                logger.info(f"  CSV 파일: {result.get('csv_path', 'N/A')}")
                all_results.append(result)
            else: # process_video_optimized 내부에서 오류 처리 후 None 반환 가능성 고려 (현재는 예외 발생 시킴)
                logger.warning(f"--- '{os.path.basename(video_folder_path_item)}' 처리 중 결과 없음 ---")

        except ValueError as ve: # process_video_optimized에서 발생할 수 있는 특정 예외
            logger.error(f"'{os.path.basename(video_folder_path_item)}' 처리 중 값 오류: {ve}")
        except Exception as e:
            logger.error(f"'{os.path.basename(video_folder_path_item)}' 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)
            # 오류 발생 시 다음 폴더로 계속 진행

    logger.info(f"=== 총 {len(all_results)}개의 비디오 폴더 처리 완료 ===")
    if all_results:
        logger.info("생성된 CSV 파일 목록:")
        for res in all_results:
            logger.info(f"  - {res.get('csv_path', 'N/A')}")

if __name__ == "__main__":
    main() 