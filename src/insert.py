import pandas as pd
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import pipe, ops
import numpy as np
from towhee.datacollection import DataCollection
import csv

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
    FieldSchema(name='id', dtype=DataType.VARCHAR, descrition='ids', max_length=500, is_primary=True, auto_id=False),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


if __name__ == "__main__":
    df = pd.read_csv('../data/raw/question_answer.csv')
    id_answer = df.set_index('id')['answer'].to_dict()

    connections.connect(host='127.0.0.1', port='19530')

    collection = create_milvus_collection('question_answer', 768)

    insert_pipe = (
        pipe.input('id', 'question', 'answer')
            .map('question', 'vec', ops.text_embedding.dpr(model_name='facebook/dpr-ctx_encoder-single-nq-base'))
            .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
            .map(('id', 'vec'), 'insert_status', ops.ann_insert.milvus_client(host='127.0.0.1', port='19530', collection_name='question_answer'))
            .output()
    )

    with open('../data/raw/question_answer.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            insert_pipe(*row)