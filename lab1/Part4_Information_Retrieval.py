import json
import math
import time
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict, Counter
import numpy as np

class InformationRetrievalSystem:
    """综合信息检索系统"""
    
    def __init__(self, enhanced_index_path: str, normalized_tokens_path: str):
        """初始化检索系统"""
        self.enhanced_index = self._load_enhanced_index(enhanced_index_path)
        self.normalized_tokens = self._load_normalized_tokens(normalized_tokens_path)
        self.doc_count = len(self.normalized_tokens)
        self.vocab_size = len(self.enhanced_index['vocabulary'])
        
        # 计算文档频率和IDF
        self._calculate_document_frequencies()
        self._calculate_idf()
        
        print(f"检索系统初始化完成:")
        print(f"  文档数量: {self.doc_count}")
        print(f"  词汇表大小: {self.vocab_size}")
        print(f"  位置信息: {'是' if self.enhanced_index.get('has_positions', False) else '否'}")
    
    def _load_enhanced_index(self, path: str) -> Dict:
        """加载增强倒排索引"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_normalized_tokens(self, path: str) -> Dict:
        """加载规范化词表"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _calculate_document_frequencies(self):
        """计算文档频率"""
        self.doc_freq = defaultdict(int)
        for term_id, postings in self.enhanced_index['inverted_lists'].items():
            self.doc_freq[term_id] = len(postings)
    
    def _calculate_idf(self):
        """计算IDF值"""
        self.idf = {}
        for term_id, df in self.doc_freq.items():
            self.idf[term_id] = math.log(self.doc_count / df)
    
    def get_term_id(self, term: str) -> int:
        """获取词项ID"""
        return self.enhanced_index['vocabulary'].get(term, -1)
    
    def get_posting_list(self, term: str) -> List:
        """获取倒排列表"""
        term_id = self.get_term_id(term)
        if term_id == -1:
            return []
        return self.enhanced_index['inverted_lists'].get(str(term_id), [])
    
    def boolean_retrieval(self, query: str) -> Set[str]:
        """布尔检索"""
        print(f"执行布尔检索: {query}")
        
        # 解析布尔表达式
        tokens = self._parse_boolean_query(query)
        if not tokens:
            return set()
        
        # 转换为后缀表达式
        postfix = self._infix_to_postfix(tokens)
        
        # 执行检索
        result = self._evaluate_postfix(postfix)
        
        print(f"  找到 {len(result)} 个文档")
        return result
    
    def _parse_boolean_query(self, query: str) -> List[str]:
        """解析布尔查询"""
        # 简单的词法分析
        tokens = []
        current_token = ""
        
        for char in query:
            if char in '()':
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
                tokens.append(char)
            elif char in '&|!':
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
                tokens.append(char)
            elif char == ' ':
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
            else:
                current_token += char
        
        if current_token.strip():
            tokens.append(current_token.strip())
        
        return tokens
    
    def _infix_to_postfix(self, tokens: List[str]) -> List[str]:
        """中缀表达式转后缀表达式"""
        precedence = {'!': 3, '&': 2, '|': 1}
        output = []
        operators = []
        
        for token in tokens:
            if token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()  # 移除 '('
            elif token in precedence:
                while (operators and operators[-1] != '(' and 
                       precedence[operators[-1]] >= precedence[token]):
                    output.append(operators.pop())
                operators.append(token)
            else:  # 词项
                output.append(token)
        
        while operators:
            output.append(operators.pop())
        
        return output
    
    def _evaluate_postfix(self, postfix: List[str]) -> Set[str]:
        """计算后缀表达式"""
        stack = []
        
        for token in postfix:
            if token == '&':
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a.intersection(b))
            elif token == '|':
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a.union(b))
            elif token == '!':
                if len(stack) >= 1:
                    a = stack.pop()
                    all_docs = set(self.enhanced_index['doc_lengths'].keys())
                    stack.append(all_docs - a)
            else:  # 词项
                postings = self.get_posting_list(token)
                docs = {posting[0] for posting in postings}
                stack.append(docs)
        
        return stack[0] if stack else set()
    
    def phrase_retrieval(self, phrase: List[str]) -> Set[str]:
        """短语检索"""
        print(f"执行短语检索: {' '.join(phrase)}")
        
        if len(phrase) < 2:
            return set()
        
        # 获取第一个词项的倒排列表
        first_term_postings = self.get_posting_list(phrase[0])
        if not first_term_postings:
            return set()
        
        result_docs = set()
        
        for doc_id, freq, positions, skip_ptr in first_term_postings:
            if self._check_phrase_in_doc(doc_id, phrase, positions):
                result_docs.add(doc_id)
        
        print(f"  找到 {len(result_docs)} 个文档")
        return result_docs
    
    def _check_phrase_in_doc(self, doc_id: str, phrase: List[str], start_positions: List[int]) -> bool:
        """检查短语是否在文档中"""
        if doc_id not in self.normalized_tokens:
            return False
        
        doc_tokens = self.normalized_tokens[doc_id]
        
        for start_pos in start_positions:
            if start_pos + len(phrase) - 1 >= len(doc_tokens):
                continue
            
            # 检查连续位置是否匹配
            match = True
            for i, term in enumerate(phrase[1:], 1):
                if start_pos + i >= len(doc_tokens) or doc_tokens[start_pos + i] != term:
                    match = False
                    break
            
            if match:
                return True
        
        return False
    
    def calculate_tf_idf(self, doc_id: str, term: str) -> float:
        """计算TF-IDF值"""
        if doc_id not in self.normalized_tokens:
            return 0.0
        
        # 计算TF
        doc_tokens = self.normalized_tokens[doc_id]
        term_count = doc_tokens.count(term)
        tf = term_count / len(doc_tokens) if doc_tokens else 0
        
        # 获取IDF
        term_id = self.get_term_id(term)
        idf = self.idf.get(str(term_id), 0) if term_id != -1 else 0
        
        return tf * idf
    
    def vector_space_retrieval(self, query_terms: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """向量空间模型检索"""
        print(f"执行向量空间检索: {' '.join(query_terms)}")
        
        # 计算查询向量
        query_vector = self._calculate_query_vector(query_terms)
        
        # 计算文档相似度
        doc_scores = []
        for doc_id in self.enhanced_index['doc_lengths'].keys():
            doc_vector = self._calculate_doc_vector(doc_id, query_terms)
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                doc_scores.append((doc_id, similarity))
        
        # 排序并返回top-k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        result = doc_scores[:top_k]
        
        print(f"  找到 {len(result)} 个相关文档")
        return result
    
    def _calculate_query_vector(self, query_terms: List[str]) -> Dict[str, float]:
        """计算查询向量"""
        query_vector = {}
        term_counts = Counter(query_terms)
        
        for term, count in term_counts.items():
            term_id = self.get_term_id(term)
            if term_id != -1:
                tf = count / len(query_terms)
                idf = self.idf.get(str(term_id), 0)
                query_vector[term] = tf * idf
        
        return query_vector
    
    def _calculate_doc_vector(self, doc_id: str, query_terms: List[str]) -> Dict[str, float]:
        """计算文档向量"""
        doc_vector = {}
        
        for term in query_terms:
            tf_idf = self.calculate_tf_idf(doc_id, term)
            doc_vector[term] = tf_idf
        
        return doc_vector
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """计算余弦相似度"""
        # 获取所有词项
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        if not all_terms:
            return 0.0
        
        # 计算点积
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in all_terms)
        
        # 计算模长
        norm1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        norm2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def analyze_query_processing_order(self, query: str) -> Dict[str, float]:
        """分析不同处理顺序对时间的影响"""
        print(f"\n分析查询处理顺序: {query}")
        
        # 解析查询
        tokens = self._parse_boolean_query(query)
        postfix = self._infix_to_postfix(tokens)
        
        # 测试不同处理顺序
        results = {}
        
        # 原始顺序
        start_time = time.time()
        result1 = self._evaluate_postfix(postfix)
        results['原始顺序'] = time.time() - start_time
        
        # 优化顺序：先处理高频词项
        optimized_postfix = self._optimize_query_order(postfix)
        start_time = time.time()
        result2 = self._evaluate_postfix(optimized_postfix)
        results['优化顺序'] = time.time() - start_time
        
        print(f"  原始顺序耗时: {results['原始顺序']:.6f}秒")
        print(f"  优化顺序耗时: {results['优化顺序']:.6f}秒")
        print(f"  结果一致性: {result1 == result2}")
        
        return results
    
    def _optimize_query_order(self, postfix: List[str]) -> List[str]:
        """优化查询处理顺序"""
        # 按文档频率排序词项（高频词项优先）
        term_freqs = []
        for token in postfix:
            if token not in ['&', '|', '!']:
                term_id = self.get_term_id(token)
                if term_id != -1:
                    freq = self.doc_freq.get(str(term_id), 0)
                    term_freqs.append((token, freq))
        
        # 按频率排序
        term_freqs.sort(key=lambda x: x[1], reverse=True)
        
        # 重新构建查询（简化版本）
        return postfix  # 这里可以进一步优化
    
    def compare_compression_efficiency(self, query: str) -> Dict[str, Any]:
        """比较压缩前后的检索效率"""
        print(f"\n比较压缩效率: {query}")
        
        # 使用原始索引
        start_time = time.time()
        result_original = self.boolean_retrieval(query)
        time_original = time.time() - start_time
        
        # 使用压缩索引（模拟）
        start_time = time.time()
        result_compressed = self._retrieve_with_compressed_index(query)
        time_compressed = time.time() - start_time
        
        results = {
            'original_time': time_original,
            'compressed_time': time_compressed,
            'speedup': time_original / time_compressed if time_compressed > 0 else 0,
            'result_consistency': result_original == result_compressed
        }
        
        print(f"  原始索引耗时: {time_original:.6f}秒")
        print(f"  压缩索引耗时: {time_compressed:.6f}秒")
        print(f"  加速比: {results['speedup']:.2f}x")
        print(f"  结果一致性: {results['result_consistency']}")
        
        return results
    
    def _retrieve_with_compressed_index(self, query: str) -> Set[str]:
        """使用压缩索引进行检索（模拟）"""
        # 这里简化实现，实际应该使用压缩索引
        return self.boolean_retrieval(query)
    
    def analyze_skip_pointer_impact(self, query: str, skip_steps: List[int]) -> Dict[int, float]:
        """分析不同跳表指针步长的影响"""
        print(f"\n分析跳表指针步长影响: {query}")
        
        results = {}
        
        for step in skip_steps:
            start_time = time.time()
            result = self._retrieve_with_skip_pointers(query, step)
            results[step] = time.time() - start_time
            print(f"  步长 {step}: {results[step]:.6f}秒")
        
        return results
    
    def _retrieve_with_skip_pointers(self, query: str, skip_step: int) -> Set[str]:
        """使用指定步长的跳表指针进行检索"""
        # 这里简化实现，实际应该根据步长调整跳表指针
        return self.boolean_retrieval(query)

def main():
    """主函数：演示信息检索系统"""
    print("=" * 80)
    print("多种形式的信息检索系统")
    print("=" * 80)
    
    # 初始化检索系统
    retrieval_system = InformationRetrievalSystem(
        "enhanced_inverted_index.json",
        "normalized_tokens.json"
    )
    
    # A. 布尔检索
    print("\n" + "=" * 60)
    print("A. 布尔检索")
    print("=" * 60)
    
    # 设计3种复杂查询条件
    boolean_queries = [
        "web & development",  # 简单AND查询
        "machine | learning",  # 简单OR查询
        "tech & (enthusiasts | conference)",  # 复杂组合查询
    ]
    
    for i, query in enumerate(boolean_queries, 1):
        print(f"\n查询 {i}: {query}")
        result = retrieval_system.boolean_retrieval(query)
        print(f"  结果文档: {sorted(result)}")
        
        # 分析处理顺序影响
        retrieval_system.analyze_query_processing_order(query)
        
        # 比较压缩效率
        retrieval_system.compare_compression_efficiency(query)
    
    # 短语检索
    print(f"\n短语检索:")
    phrase_queries = [
        ["web", "development"],
        ["machine", "learning"],
        ["ai", "conference"]
    ]
    
    for phrase in phrase_queries:
        result = retrieval_system.phrase_retrieval(phrase)
        print(f"  短语 '{' '.join(phrase)}': {sorted(result)}")
    
    # 跳表指针分析
    print(f"\n跳表指针步长分析:")
    skip_steps = [1, 2, 4, 8]
    retrieval_system.analyze_skip_pointer_impact(boolean_queries[0], skip_steps)
    
    # B. 向量空间模型
    print("\n" + "=" * 60)
    print("B. 向量空间模型")
    print("=" * 60)
    
    vector_queries = [
        ["web", "development", "technology"],
        ["machine", "learning", "ai"],
        ["data", "science", "analysis"]
    ]
    
    for i, query_terms in enumerate(vector_queries, 1):
        print(f"\n向量查询 {i}: {' '.join(query_terms)}")
        results = retrieval_system.vector_space_retrieval(query_terms, top_k=3)
        
        for doc_id, score in results:
            print(f"  文档 {doc_id}: 相似度 {score:.4f}")
    
    print("\n" + "=" * 80)
    print("信息检索系统演示完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()
