import pandas as pd
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import pipe, ops
import numpy as np
from towhee.datacollection import DataCollection
import csv

def ann_search(question):
    connections.connect(host='127.0.0.1', port='19530')

    df = pd.read_csv('../data/raw/question_answer.csv')
    id_answer = df.set_index('id')['answer'].to_dict()

    collection = Collection(name='question_answer')
    
    ans_pipe = (
        pipe.input('question')
            .map('question', 'vec', ops.text_embedding.dpr(model_name="facebook/dpr-ctx_encoder-single-nq-base"))
            .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
            .map('vec', 'res', ops.ann_search.milvus_client(host='127.0.0.1', port='19530', collection_name='question_answer', limit=10))
            .map('res', 'answer', lambda x: [id_answer[int(i[0])] for i in x])
            .output('question', 'answer')
    )

    ans = ans_pipe(question)
    ans = DataCollection(ans)
    return ans[0]['answer']