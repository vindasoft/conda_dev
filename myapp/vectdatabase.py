import chromadb
from sentence_transformers import SentenceTransformer
from typing import List

# 声明文本转向量模型存放路径
text2vec_path = "F:\\python_model\\text2vec-base-chinese"

# 声明外部模型存放路径
cross_encoder_path = "F:\\python_model\\mmarco-mMiniLMv2-L12-H384-v1"

# 自定义函数：将文件内容做分片处理，方便向量化（防止文件内容过长，导致向量化失败）
def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, 'r') as file:
        content = file.read()
    return [chunk for chunk in content.split("\n\n")]

# 将文件内容做分片处理
chunks = split_into_chunks("D:\\python_dev\\conda_dev\\doctest.txt")

# 指定模型路径并加载模型
embedding_model = SentenceTransformer(text2vec_path)

# 自定义函数：文本转向量，返回向量列表
def embed_chunk(chunk: str) -> List[float]:
    # 将文本参数做向量化处理
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    # 返回向量列表
    return embedding.tolist()

chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")

# 调用文本转向量函数，获取转换后的向量列表
embeddings = [embed_chunk(chunk) for chunk in chunks]

'''
 * 自定义函数save_embeddings：保存向量列表数据至数据库
 * chunks:是原始文本分片内容列表
 * embeddings：是原始文件分片内容做向量化转换后的内容列表
 * ids:是列表索引序号
'''
def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents = [chunk],
            embeddings = [embedding],
            ids = [str(i)]
        )

# 调用保持向量内容至向量数据库chromadb
save_embeddings(chunks, embeddings)

'''
  # 定义数据库检索函数
  # 参数：query-检索的内容，top_k返回TopK的内容
'''
def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k
    )
    return results['documents'][0]

#query = "用户需求是什么？“
query = "用户需求"
retrieve_chunks = retrieve(query, 2)

'''
for i, chunk in enumerate(retrieve_chunks):
    print(f"[{i}] {chunk}\n")
'''

# 如果数据库检索的结果不够理想，还可以调用外部模型进行检索
from sentence_transformers import CrossEncoder

# 定义外部调用接口：将检索结果列表（内容：是按照相似度得分降序排序），取topk的内容返回。
def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    cross_encoder = CrossEncoder(cross_encoder_path)
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # 这是一个列表推导，遍历 scored_chunks 中的每一个元组
    # 使用 chunk, _ 对元组进行解包：将元组的第一个元素赋值给 chunk，第二个元素赋值给 _。
    # _ 是一个约定俗成的变量名，表示“我们不关心这个值”（这里是得分），仅用于占位，避免语法错误
    # 结果是一个新列表，只包含每个元组中的第一个元素，即所有文本块（保持原有的顺序）。
    return [chunk for chunk, _ in scored_chunks][:top_k]

# 调用外部模型的检索接口：返回列表
reranked_chunks = rerank(query, retrieve_chunks, 3)

for i, chunk in enumerate(reranked_chunks):
    print(f"[{i}] {chunk}\n")
