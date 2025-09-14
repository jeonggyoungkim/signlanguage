# coding: utf-8
import numpy as np
import os
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset

# 전역 상수: 특수 토큰 정의
SIL_TOKEN = "<si>" # 침묵(silence) 또는 공백(blank) 토큰, CTC Loss에서 사용될 수 있음
UNK_TOKEN = "<unk>" # 어휘 사전에 없는 토큰 (unknown token)
PAD_TOKEN = "<pad>" # 시퀀스 길이를 맞추기 위한 패딩 토큰
BOS_TOKEN = "<s>"   # 문장 시작 (beginning-of-sentence) 토큰
EOS_TOKEN = "</s>"  # 문장 끝 (end-of-sentence) 토큰


class Vocabulary:
    """ 어휘 사전 클래스. 토큰과 인덱스 간의 매핑을 나타냅니다. """

    def __init__(self):
        self.specials = [] # 특수 토큰 리스트
        self.itos = [] # 인덱스를 토큰으로 변환 (index-to-string)
        self.unk_index = 0  # UNK_TOKEN의 기본 인덱스, 나중에 업데이트됨
        self.stoi = defaultdict(self._get_unk_index) # 람다 대신 메소드 참조
        self.unk_token = UNK_TOKEN
        self.pad_token = PAD_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN

    def _get_unk_index(self): # defaultdict의 factory로 사용될 메소드
        return self.unk_index

    def _from_list(self, tokens: List[str] = None):
        """
        토큰 리스트로부터 어휘 사전을 생성합니다.
        토큰들은 고유하며 미리 선택된 것으로 가정합니다.
        특수 토큰이 리스트에 없으면 추가됩니다.

        :param tokens: 토큰 리스트
        """
        # 먼저 specials를 포함한 모든 초기 토큰을 itos와 stoi에 추가
        # 이 과정에서 unk_token의 실제 인덱스가 결정됨
        current_tokens = self.specials + (tokens or [])
        self.itos = [] # 초기화
        self.stoi = defaultdict(self._get_unk_index) # 초기화 (새로운 unk_index를 참조하기 위해)
        
        for t in current_tokens:
            if t not in self.itos: # 중복 방지
                new_idx = len(self.itos)
                self.itos.append(t)
                self.stoi[t] = new_idx
        
        # UNK_TOKEN의 실제 인덱스를 self.unk_index에 저장 (중요!)
        if self.unk_token in self.stoi:
            self.unk_index = self.stoi[self.unk_token]
        else:
            # UNK_TOKEN이 초기 토큰 목록에 없으면, 강제로 추가하고 unk_index 설정
            if self.unk_token not in self.itos:
                unk_actual_idx = len(self.itos)
                self.itos.append(self.unk_token)
                self.stoi[self.unk_token] = unk_actual_idx
                self.unk_index = unk_actual_idx
            else: # 이미 itos에 있지만 stoi에 없는 경우는 거의 없음 (버그 상황)
                self.unk_index = self.stoi[self.unk_token] # 이미 있는 경우

        # defaultdict가 올바른 unk_index를 참조하도록 다시 설정 (필수는 아닐 수 있으나 안전을 위해)
        # self.stoi.default_factory = self._get_unk_index 
        # 위 라인은 defaultdict 객체를 새로 만들지 않으면 효과가 없을 수 있음.
        # _from_list 시작 시 stoi를 새로 defaultdict(self._get_unk_index)로 만드는 것으로 충분할 수 있음.

        assert len(self.stoi) == len(self.itos)

    def _from_file(self, file: str):
        """
        파일 내용으로부터 어휘 사전을 생성합니다.
        파일 형식: i번째 인덱스의 토큰은 i번째 줄에 있습니다.

        :param file: 어휘 사전을 불러올 파일 경로
        """
        tokens = []
        with open(file, "r", encoding="utf-8") as open_file:
            for line in open_file:
                l = line.strip("\n").split('\t') # 탭으로 분리된 경우 처리 (예: 토큰 ID와 토큰)
                if len(l) > 1:
                    tokens.append(l[1]) # 두 번째 요소 (토큰) 사용
                else:
                    tokens.append(line.strip("\n")) # 줄 전체를 토큰으로 사용
        self._from_list(tokens)

    def __str__(self) -> str:
        # stoi가 defaultdict이므로, 일반 dict처럼 출력하기 위해 변환
        return str(dict(self.stoi))

    def to_file(self, file: str):
        """
        어휘 사전을 파일에 저장합니다. i번째 인덱스의 토큰을 i번째 줄에 씁니다.

        :param file: 어휘 사전을 저장할 파일 경로
        """
        with open(file, "w", encoding="utf-8") as open_file:
            for t in self.itos:
                open_file.write("{}\n".format(t))

    def add_tokens(self, tokens: List[str]):
        """
        어휘 사전에 토큰 리스트를 추가합니다.
        이 메소드는 _from_list 내부 로직과 유사하므로, _from_list 사용을 권장.
        만약 직접 사용한다면, unk_index 업데이트에 주의해야 함.

        :param tokens: 어휘 사전에 추가할 토큰 리스트
        """
        for t in tokens:
            if t not in self.itos:
                new_index = len(self.itos)
                self.itos.append(t)
                self.stoi[t] = new_index
        # add_tokens 후에도 self.unk_index가 정확한지 확인/업데이트 필요할 수 있음
        if self.unk_token in self.stoi:
            self.unk_index = self.stoi[self.unk_token]

    def is_unk(self, token: str) -> bool:
        """
        토큰이 어휘 사전에 포함되어 있는지 (UNK 토큰인지) 확인합니다.
        defaultdict의 동작으로 인해, 없는 토큰은 unk_index로 매핑됨.

        :param token: 확인할 토큰
        :return: 포함되어 있으면 True, 아니면 False
        """
        return self.stoi[token] == self.unk_index and token != self.unk_token

    def __len__(self) -> int:
        return len(self.itos)


class TextVocabulary(Vocabulary):
    def __init__(self, cfg: Dict[str, Any], tokens: List[str] = None, file: str = None):
        """
        토큰 리스트나 파일로부터 어휘 사전을 생성합니다.

        특수 토큰이 파일이나 리스트에 이미 없으면 추가됩니다.
        파일 형식: i번째 인덱스의 토큰은 i번째 줄에 있습니다.

        :param cfg: 설정 객체
        :param tokens: 토큰 리스트
        :param file: 어휘 사전을 불러올 파일
        """
        super().__init__()
        self.specials = [self.unk_token, self.pad_token, self.bos_token, self.eos_token]

        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

    def array_to_sentence(self, array: np.array, cut_at_eos=True) -> List[str]:
        """
        ID 배열을 문장(토큰 리스트)으로 변환합니다. 선택적으로 EOS 토큰에서 자를 수 있습니다.

        :param array: 인덱스를 포함하는 1D 배열
        :param cut_at_eos: 첫 번째 <eos> 토큰에서 디코딩된 문장을 자를지 여부
        :return: 문자열(토큰) 리스트
        """
        sentence = []
        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == self.eos_token:
                break
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.array, cut_at_eos=True) -> List[List[str]]:
        """
        여러 개의 토큰 ID 시퀀스 배열을 문장 리스트로 변환합니다.
        선택적으로 EOS 토큰에서 자를 수 있습니다.

        :param arrays: 인덱스를 포함하는 2D 배열
        :param cut_at_eos: 첫 번째 <eos> 토큰에서 디코딩된 문장을 자를지 여부
        :return: 문자열(토큰) 리스트의 리스트
        """
        sentences = []
        for array in arrays:
            sentences.append(self.array_to_sentence(array=array, cut_at_eos=cut_at_eos))
        return sentences


class GlossVocabulary(Vocabulary):
    def __init__(self, tokens: List[str] = None, file: str = None):
        """
        토큰 리스트나 파일로부터 어휘 사전을 생성합니다.

        특수 토큰이 파일이나 리스트에 이미 없으면 추가됩니다.
        파일 형식: i번째 인덱스의 토큰은 i번째 줄에 있습니다.

        :param tokens: 토큰 리스트
        :param file: 어휘 사전을 불러올 파일
        """
        super().__init__()
        self.specials = [SIL_TOKEN, self.unk_token, self.pad_token]

        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

        assert self.stoi[SIL_TOKEN] == 0 # SIL_TOKEN의 인덱스가 0인지 확인 (CTC Loss blank 인덱스와 일치시키기 위함)

    def array_to_sentence(self, array: np.array) -> List[str]:
        """ ID 배열을 Gloss 시퀀스(토큰 리스트)로 변환합니다. """
        sequence = []
        for i in array:
            sequence.append(self.itos[i])
        return sequence

    def arrays_to_sentences(self, arrays: np.array) -> List[List[str]]:
        """ 여러 ID 배열을 Gloss 시퀀스 리스트로 변환합니다. """
        gloss_sequences = []
        for array in arrays:
            gloss_sequences.append(self.array_to_sentence(array))
        return gloss_sequences


def filter_min(counter: Counter, minimum_freq: int):
    """ 최소 빈도수로 카운터를 필터링합니다. """
    filtered_counter = Counter({t: c for t, c in counter.items() if c >= minimum_freq})
    return filtered_counter


def sort_and_cut(counter: Counter, limit: int):
    """ 카운터를 가장 빈번한 토큰 순으로 정렬하고, 지정된 수만큼 자릅니다.
        빈도수가 같을 경우 알파벳 순으로 정렬합니다. """
    # 빈도수로 먼저 정렬 (내림차순), 그 다음 알파벳 순 (오름차순)으로 정렬
    tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0]) # 1. 알파벳 순 정렬
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True) # 2. 빈도수 기준 내림차순 정렬
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]] # 지정된 limit만큼 선택
    return vocab_tokens


def build_vocab(
    cfg: Dict[str, Any], 
    field: str, 
    max_size: int, 
    min_freq: int, 
    dataset: Dataset, 
    vocab_file: str = None
) -> Vocabulary:
    """
    주어진 `dataset`이나 `vocab_file`로부터 어휘 사전을 구축합니다.

    :param cfg: 설정 객체
    :param field: 어휘 사전을 구축할 대상 필드 이름 (예: "gls", "txt")
    :param max_size: 어휘 사전의 최대 크기
    :param min_freq: 토큰이 사전에 포함되기 위한 최소 등장 빈도
    :param dataset: 어휘 사전을 구축할 데이터셋
    :param vocab_file: (선택 사항) 미리 정의된 어휘 사전 파일 경로
    :return: 구축된 어휘 사전 객체
    """
    # vocab_file이 제공되고 존재하면, 해당 파일에서 어휘 사전을 로드합니다.
    if vocab_file is not None and os.path.exists(vocab_file):
        if field == "gls":
            vocab = GlossVocabulary(file=vocab_file)
        else:
            raise ValueError(f"Unknown field type for vocab building: {field}")
    else:
        # vocab_file이 없으면 데이터셋에서 어휘 사전을 구축합니다.
        tokens_from_data = []
        for ex in dataset:
            if field in ex:
                tokens_from_data.extend(ex[field])
        
        counter = Counter(tokens_from_data)
        # 최소 빈도수 필터링
        if min_freq > 1:
            counter = filter_min(counter, min_freq)
        # 최대 크기 제한 및 정렬
        vocab_tokens = sort_and_cut(counter, max_size)

        if field == "gls":
            vocab = GlossVocabulary(tokens=vocab_tokens)
        else:
            raise ValueError(f"Unknown field type for vocab building: {field}")

    return vocab
