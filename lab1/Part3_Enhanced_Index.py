import math
import json
import struct
from typing import List, Tuple, Dict, Any, Set
from collections import defaultdict

class EnhancedInvertedIndex:
    """
    增强的倒排索引类，支持位置信息和压缩功能
    """
    
    def __init__(self):
        # 基础结构
        self.vocabulary = {}           # 词汇表：词项 -> 词项ID
        self.inverted_lists = {}       # 倒排列表：词项ID -> [(文档ID, 词频, 位置列表, 跳表指针), ...]
        self.doc_lengths = {}          # 文档长度：文档ID -> 文档中的词项数量
        self.doc_sort_keys = {}        # 文档ID到排序键的映射
        self.sort_key_to_doc = {}      # 排序键到文档ID的映射
        self.next_sort_key = 0
        
        # 压缩相关
        self.block_size = 4            # 按块存储的块大小
        self.compressed_vocabulary = {} # 压缩后的词汇表
        self.compressed_lists = {}     # 压缩后的倒排列表
    
    def load_from_basic_index(self, basic_index_path: str, normalized_tokens_path: str):
        """
        从基础倒排索引和规范化词表加载数据，并添加位置信息
        
        参数:
            basic_index_path: 基础倒排索引文件路径
            normalized_tokens_path: 规范化词表文件路径
        """
        print("从基础索引加载数据并添加位置信息...")
        
        # 加载基础索引
        with open(basic_index_path, 'r', encoding='utf-8') as f:
            basic_data = json.load(f)
        
        self.vocabulary = basic_data['vocabulary']
        self.doc_lengths = basic_data['doc_lengths']
        self.doc_sort_keys = basic_data['doc_sort_keys']
        self.sort_key_to_doc = basic_data['sort_key_to_doc']
        self.next_sort_key = basic_data['next_sort_key']
        
        # 加载原始词表以获取位置信息
        with open(normalized_tokens_path, 'r', encoding='utf-8') as f:
            self.normalized_tokens = json.load(f)
        
        # 构建带位置信息的倒排列表
        self._build_positional_index()
        
        print(f"位置信息添加完成：{len(self.inverted_lists)} 个词项")
    
    def _build_positional_index(self):
        """构建带位置信息的倒排索引"""
        print("构建带位置信息的倒排索引...")
        
        # 为每个词项构建位置信息
        for term, term_id in self.vocabulary.items():
            self.inverted_lists[term_id] = []
            
            # 收集该词项在所有文档中的位置
            term_positions = defaultdict(list)  # 文档ID -> 位置列表
            
            for doc_id, tokens in self.normalized_tokens.items():
                positions = []
                for i, token in enumerate(tokens):
                    if token == term:
                        positions.append(i)
                
                if positions:  # 如果该文档包含该词项
                    term_positions[doc_id] = positions
            
            # 构建倒排记录
            for doc_id, positions in term_positions.items():
                freq = len(positions)
                skip_ptr = -1  # 暂时不设置跳表指针
                self.inverted_lists[term_id].append((doc_id, freq, positions, skip_ptr))
            
            # 按文档排序
            self.inverted_lists[term_id].sort(key=lambda x: self.doc_sort_keys[x[0]])
    
    def add_skip_pointers(self):
        """为长倒排列表添加跳表指针"""
        print("添加跳表指针...")
        skip_count = 0
        
        for term_id, postings in self.inverted_lists.items():
            if len(postings) >= 4:
                n = len(postings)
                skip_interval = int(math.sqrt(n))
                
                for i in range(0, n, skip_interval):
                    if i + skip_interval < n:
                        doc_id, freq, positions, _ = postings[i]
                        postings[i] = (doc_id, freq, positions, i + skip_interval)
                        skip_count += 1
        
        print(f"添加了 {skip_count} 个跳表指针")
    
    def implement_blocking_compression(self):
        """
        实现按块存储压缩
        将连续的k个词项存储在一个块中，减少指针数量
        """
        print(f"实现按块存储压缩 (块大小: {self.block_size})...")
        
        self.compressed_vocabulary = {}
        self.compressed_lists = {}
        
        # 按词项ID排序
        sorted_terms = sorted(self.vocabulary.items(), key=lambda x: x[1])
        
        # 按块处理词项
        for i in range(0, len(sorted_terms), self.block_size):
            block_terms = sorted_terms[i:i + self.block_size]
            block_id = i // self.block_size
            
            # 创建块
            block_data = {
                'terms': [term for term, _ in block_terms],
                'term_ids': [term_id for _, term_id in block_terms],
                'postings': []
            }
            
            # 合并块内所有词项的倒排列表
            for term, term_id in block_terms:
                if term_id in self.inverted_lists:
                    for posting in self.inverted_lists[term_id]:
                        block_data['postings'].append((term_id, posting))
            
            # 按文档排序
            block_data['postings'].sort(key=lambda x: self.doc_sort_keys[x[1][0]])
            
            self.compressed_lists[block_id] = block_data
            
            # 更新词汇表映射
            for term, term_id in block_terms:
                self.compressed_vocabulary[term] = {
                    'block_id': block_id,
                    'local_id': term_id % self.block_size
                }
        
        print(f"按块存储完成：{len(self.compressed_lists)} 个块")
    
    def implement_front_coding_compression(self):
        """
        实现前端编码压缩
        利用连续词项的共同前缀进行压缩
        """
        print("实现前端编码压缩...")
        
        # 按字母顺序排序词项
        sorted_terms = sorted(self.vocabulary.keys())
        
        self.front_coded_vocabulary = {}
        self.front_coded_terms = []
        
        i = 0
        while i < len(sorted_terms):
            current_term = sorted_terms[i]
            
            # 寻找与当前词项有共同前缀的后续词项
            prefix_length = 0
            j = i + 1
            
            while j < len(sorted_terms):
                next_term = sorted_terms[j]
                common_prefix = self._get_common_prefix(current_term, next_term)
                
                if common_prefix > 0:
                    prefix_length = common_prefix
                    j += 1
                else:
                    break
            
            # 存储压缩后的词项
            if prefix_length > 0:
                # 有共同前缀的情况
                prefix = current_term[:prefix_length]
                suffix = current_term[prefix_length:]
                
                compressed_entry = {
                    'type': 'prefix',
                    'prefix': prefix,
                    'suffixes': [suffix]
                }
                
                # 添加同前缀的其他词项
                for k in range(i + 1, j):
                    term = sorted_terms[k]
                    suffix = term[prefix_length:]
                    compressed_entry['suffixes'].append(suffix)
                
                self.front_coded_terms.append(compressed_entry)
                
                # 更新词汇表
                for k in range(i, j):
                    term = sorted_terms[k]
                    self.front_coded_vocabulary[term] = {
                        'entry_index': len(self.front_coded_terms) - 1,
                        'suffix_index': k - i
                    }
                
                i = j
            else:
                # 无共同前缀的情况
                compressed_entry = {
                    'type': 'single',
                    'term': current_term
                }
                self.front_coded_terms.append(compressed_entry)
                
                self.front_coded_vocabulary[current_term] = {
                    'entry_index': len(self.front_coded_terms) - 1,
                    'suffix_index': 0
                }
                
                i += 1
        
        print(f"前端编码完成：{len(self.front_coded_terms)} 个压缩条目")
    
    def _get_common_prefix(self, term1: str, term2: str) -> int:
        """计算两个词项的共同前缀长度"""
        min_len = min(len(term1), len(term2))
        for i in range(min_len):
            if term1[i] != term2[i]:
                return i
        return min_len
    
    def search_phrase(self, phrase_terms: List[str]) -> Dict[str, Any]:
        """
        短语检索：查找包含指定词序的文档
        
        参数:
            phrase_terms: 短语中的词项列表
            
        返回:
            包含短语的文档列表
        """
        print(f"短语检索: {' '.join(phrase_terms)}")
        
        if not phrase_terms:
            return {"documents": [], "message": "空短语"}
        
        # 获取第一个词项的倒排列表
        first_term = phrase_terms[0]
        if first_term not in self.vocabulary:
            return {"documents": [], "message": f"词项 '{first_term}' 不存在"}
        
        first_term_id = self.vocabulary[first_term]
        if first_term_id not in self.inverted_lists:
            return {"documents": [], "message": f"词项 '{first_term}' 无倒排列表"}
        
        # 从第一个词项开始，逐步验证短语
        candidate_docs = []
        
        for doc_id, freq, positions, skip_ptr in self.inverted_lists[first_term_id]:
            # 检查该文档是否包含完整的短语
            if self._check_phrase_in_doc(doc_id, phrase_terms, positions):
                candidate_docs.append(doc_id)
        
        return {
            "documents": candidate_docs,
            "phrase": ' '.join(phrase_terms),
            "result_count": len(candidate_docs)
        }
    
    def _check_phrase_in_doc(self, doc_id: str, phrase_terms: List[str], start_positions: List[int]) -> bool:
        """检查文档中是否包含指定短语"""
        if doc_id not in self.normalized_tokens:
            return False
        
        tokens = self.normalized_tokens[doc_id]
        
        # 对每个起始位置，检查是否形成完整短语
        for start_pos in start_positions:
            if start_pos + len(phrase_terms) > len(tokens):
                continue
            
            # 检查从start_pos开始的词序是否匹配短语
            match = True
            for i, term in enumerate(phrase_terms):
                if tokens[start_pos + i] != term:
                    match = False
                    break
            
            if match:
                return True
        
        return False
    
    def calculate_storage_sizes(self) -> Dict[str, int]:
        """计算各种格式的存储大小"""
        sizes = {}
        
        # 原始格式大小
        original_data = {
            "vocabulary": self.vocabulary,
            "inverted_lists": self.inverted_lists,
            "doc_lengths": self.doc_lengths,
            "doc_sort_keys": self.doc_sort_keys,
            "sort_key_to_doc": self.sort_key_to_doc,
            "next_sort_key": self.next_sort_key
        }
        
        original_json = json.dumps(original_data, ensure_ascii=False)
        sizes['original'] = len(original_json.encode('utf-8'))
        
        # 按块存储大小
        if hasattr(self, 'compressed_lists'):
            blocking_data = {
                "compressed_vocabulary": self.compressed_vocabulary,
                "compressed_lists": self.compressed_lists,
                "block_size": self.block_size
            }
            blocking_json = json.dumps(blocking_data, ensure_ascii=False)
            sizes['blocking'] = len(blocking_json.encode('utf-8'))
        
        # 前端编码大小
        if hasattr(self, 'front_coded_vocabulary'):
            front_coding_data = {
                "front_coded_vocabulary": self.front_coded_vocabulary,
                "front_coded_terms": self.front_coded_terms
            }
            front_coding_json = json.dumps(front_coding_data, ensure_ascii=False)
            sizes['front_coding'] = len(front_coding_json.encode('utf-8'))
        
        return sizes
    
    def save_enhanced_index(self, output_path: str):
        """保存增强的倒排索引（紧凑格式）"""
        print(f"保存增强倒排索引到: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('{\n')
                
                # 保存基础信息
                f.write('  "vocabulary": ')
                json.dump(self.vocabulary, f, indent=2, ensure_ascii=False, separators=(',', ': '))
                f.write(',\n')
                
                # 保存倒排列表（紧凑格式）
                f.write('  "inverted_lists": {\n')
                term_ids = sorted(self.inverted_lists.keys(), key=int)
                for i, term_id in enumerate(term_ids):
                    f.write(f'    "{term_id}": [')
                    postings = self.inverted_lists[term_id]
                    for j, posting in enumerate(postings):
                        if j > 0:
                            f.write(', ')
                        # 格式化四元组：[文档ID, 词频, 位置列表, 跳表指针]
                        doc_id, freq, positions, skip_ptr = posting
                        f.write(f'["{doc_id}", {freq}, {json.dumps(positions)}, {skip_ptr}]')
                    f.write(']')
                    if i < len(term_ids) - 1:
                        f.write(',')
                    f.write('\n')
                f.write('  },\n')
                
                # 保存其他信息
                f.write('  "doc_lengths": ')
                json.dump(self.doc_lengths, f, indent=2, ensure_ascii=False, separators=(',', ': '))
                f.write(',\n')
                
                f.write('  "doc_sort_keys": ')
                json.dump(self.doc_sort_keys, f, indent=2, ensure_ascii=False, separators=(',', ': '))
                f.write(',\n')
                
                f.write('  "sort_key_to_doc": ')
                json.dump(self.sort_key_to_doc, f, indent=2, ensure_ascii=False, separators=(',', ': '))
                f.write(',\n')
                
                f.write(f'  "next_sort_key": {self.next_sort_key},\n')
                f.write('  "has_positions": true,\n')
                
                # 保存压缩方法
                compression_methods = []
                
                if hasattr(self, 'compressed_lists'):
                    f.write('  "blocking_compression": {\n')
                    f.write('    "compressed_vocabulary": ')
                    json.dump(self.compressed_vocabulary, f, indent=4, ensure_ascii=False, separators=(',', ': '))
                    f.write(',\n')
                    f.write('    "compressed_lists": {\n')
                    for j, (block_id, block_data) in enumerate(self.compressed_lists.items()):
                        f.write(f'      "{block_id}": {{\n')
                        f.write('        "terms": ')
                        json.dump(block_data['terms'], f, ensure_ascii=False, separators=(',', ': '))
                        f.write(',\n')
                        f.write('        "term_ids": ')
                        json.dump(block_data['term_ids'], f, ensure_ascii=False, separators=(',', ': '))
                        f.write(',\n')
                        f.write('        "postings": [')
                        for k, (term_id, posting) in enumerate(block_data['postings']):
                            if k > 0:
                                f.write(', ')
                            # 格式化压缩后的倒排记录：[词项ID, [文档ID, 词频, 位置列表, 跳表指针]]
                            doc_id, freq, positions, skip_ptr = posting
                            f.write(f'[{term_id}, ["{doc_id}", {freq}, {json.dumps(positions)}, {skip_ptr}]]')
                        f.write(']\n')
                        f.write('      }')
                        if j < len(self.compressed_lists) - 1:
                            f.write(',')
                        f.write('\n')
                    f.write('    },\n')
                    f.write(f'    "block_size": {self.block_size}\n')
                    f.write('  },\n')
                    compression_methods.append("blocking")
                
                if hasattr(self, 'front_coded_vocabulary'):
                    f.write('  "front_coding_compression": {\n')
                    f.write('    "front_coded_vocabulary": ')
                    json.dump(self.front_coded_vocabulary, f, indent=4, ensure_ascii=False, separators=(',', ': '))
                    f.write(',\n')
                    f.write('    "front_coded_terms": ')
                    json.dump(self.front_coded_terms, f, indent=4, ensure_ascii=False, separators=(',', ': '))
                    f.write('\n')
                    f.write('  },\n')
                    compression_methods.append("front_coding")
                
                f.write('  "compression_methods": ')
                json.dump(compression_methods, f, ensure_ascii=False)
                f.write('\n')
                f.write('}\n')
                
            print("增强倒排索引保存成功")
        except Exception as e:
            print(f"保存失败: {e}")
    
    def print_compression_statistics(self):
        """打印压缩统计信息"""
        sizes = self.calculate_storage_sizes()
        
        print("\n压缩统计信息:")
        print(f"原始格式大小: {sizes['original']:,} 字节")
        
        if 'blocking' in sizes:
            blocking_ratio = sizes['blocking'] / sizes['original']
            print(f"按块存储大小: {sizes['blocking']:,} 字节 (压缩比: {blocking_ratio:.2%})")
        
        if 'front_coding' in sizes:
            front_coding_ratio = sizes['front_coding'] / sizes['original']
            print(f"前端编码大小: {sizes['front_coding']:,} 字节 (压缩比: {front_coding_ratio:.2%})")
        
        if 'blocking' in sizes and 'front_coding' in sizes:
            combined_ratio = sizes['front_coding'] / sizes['original']
            print(f"最佳压缩比: {combined_ratio:.2%}")

def main():
    """主函数：构建增强倒排索引（位置信息与压缩优化）"""
    print("=" * 60)
    print("增强倒排索引系统 - 位置信息与压缩优化")
    print("=" * 60)
    
    # 创建增强索引实例
    index = EnhancedInvertedIndex()
    
    # 加载基础索引并添加位置信息
    index.load_from_basic_index("inverted_index.json", "normalized_tokens.json")
    
    # 添加跳表指针
    index.add_skip_pointers()
    
    # 实现压缩方法
    print("\n" + "=" * 40)
    print("实现压缩方法")
    print("=" * 40)
    
    index.implement_blocking_compression()
    index.implement_front_coding_compression()
    
    # 打印压缩统计
    index.print_compression_statistics()
    
    # 保存增强索引
    print("\n" + "=" * 40)
    print("保存增强倒排索引")
    print("=" * 40)
    index.save_enhanced_index("enhanced_inverted_index.json")
    
    print("\n增强倒排索引构建完成!")

if __name__ == "__main__":
    main()
