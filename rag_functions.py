import os
import csv
import json
import numpy
import pandas as pd


def load_texts(folder_path):
    """Loads .txt files from given directory.
    Args:
        folder_path (str): path to text to be processed
    Returns:
        list: list of dictionaires with "filename" and "text" keys for every file
    """
    texts = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            text_data = {}
            file = os.path.join(folder_path, filename)
            filename = os.path.splitext(filename)[0]
            
            with open(file, "r") as f:
                text_data["filename"] = filename
                text_data["text"] = f.read()
            texts.append(text_data)
            
    return texts


def chunk_texts(texts, chunk_size, overlap):
    """Splits text into chunks of a given length (count of words).
    Args:
        filename (str): name of the text file
        text (str): text to be processed
        chunk_size (int): count of words in every text chunk
        overlap (int): overlap of words at the end of the text and the beginning of another
    Returns:
        dictionary: chunk_name as key (filename + _ + order rank of chunk for given text) and text chunk as value for every text
    """
    text_chunks = {}
    chunk_counts = []
    for text_data in texts:
        filename = text_data["filename"]
        text = text_data["text"].split()
        chunk_id = 0
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_id += 1
            chunk = " ".join(text[i : i + chunk_size])
            chunk_name = f"{filename}_{chunk_id}"
            text_chunks[chunk_name] = chunk
            
            
        chunk_counts.append(chunk_id)
    return chunk_counts, text_chunks


def split_dict_data(data):
    names = list(data)
    texts = list(data.values())
    return names, texts


def convert_questions_dict(questions):
    converted = {}
    for q in questions:
        converted[q["filename"]] = q["text"]
    return converted


def preprocess_text(texts):
    """Tokenizes every text (or chunk) with Spacy, appends tokens that are not stopwords and 
    are alphanumeric to a new list. Every list is appended to the list of all data.
    Args:
        texts (list): accepts list of strings (texts, chunks)
    Returns:
        list: list of lists of tokens
    """
    preprocessed = []
    for text in texts:
        tokens = []
        doc = nlp(text)
        for token in doc:
            if token.is_alpha and not token.is_stop:
                tokens.append(token.lemma_.lower())
        preprocessed.append(tokens)
    return preprocessed


def embed_texts(preprocessed_texts, model):
    embeddings = []
    for text in preprocessed_texts:
        text_embeddings = []
        for token in text:
            embedding = model.get_word_vector(token)
            text_embeddings.append(embedding)
        if len(text_embeddings) > 0:
            text_embeddings = numpy.mean(text_embeddings, axis=0)
        else:
            text_embeddings = numpy.zeros(300)
        embeddings.append(text_embeddings)
    return numpy.array(embeddings)



def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=5)
        
        
def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def load_right_answers(path):
    text = read_txt(path).split("\n")
    sorted_right_answers = {}
    for number, answer_name in enumerate(text, start=1):
        sorted_right_answers[number] = answer_name
    return sorted_right_answers


def get_rank_table(answers, right_answers, n):
    rank_table = []
    for entry in answers:
        sorted_chunk_names = [answer["chunk_name"] for answer in entry["returned_answers"]]
        rank_table.append(sorted_chunk_names)
    rank_table_df = pd.DataFrame(rank_table, index=list(right_answers.values()), columns=range(1,n+1))
    return rank_table_df


def get_match_count(table):
    return dict((table == numpy.array(table.index)[:, None]).sum(axis=0))


def get_top_accuracies(results, questions): # accepts only 5 results! --> needs modification if other n results is needed
    counts = list(results.values())
    cumulative_sums = []
    for k in (1, 3, 5):
        result = sum(counts[:k]) / len(questions)
        cumulative_sums.append(result)
    return cumulative_sums


def make_similarity_table(chunks_data, questions, similarity_matrix):
    rows = [name for name in chunks_data.keys()]
    table = pd.DataFrame(similarity_matrix, index=rows, columns=questions)
    return table


def mask_similarity_table(df, n):
    ranked = df.rank(ascending=False, method="first").astype(int)
    masked = ranked.where(cond=ranked <= n, other=0)
    return masked


def get_top_n_answers(chunks_data, questions, df, n):
    qna = []
    masked_df = mask_similarity_table(df, n)
    masked_dict = masked_df.to_dict()
    
    for question_name, chunk_with_rank in masked_dict.items():
        answer_chunks = []
        for chunk_name, rank in chunk_with_rank.items():
            if rank != 0:
                answer_data = {
                    "rank" : rank,
                    "chunk_name" : chunk_name,
                    "chunk_text" : chunks_data[chunk_name]
                }
                answer_chunks.append(answer_data)
        answer_chunks.sort(key=lambda x: x["rank"])
        answers = {
            "question_name" : question_name,
            "question" : questions[question_name],
            "returned_answers" : answer_chunks
        }
        qna.append(answers)  
        
    return qna

     
def get_top_n_from_db(outputs, question_names, question_texts, n):
    qna = []
    ranks = range(1, n + 1)
    for i in range(len(question_texts)):
        answer_chunks = []
        question = question_texts[i]
        chunk_names = outputs["ids"][i]
        chunk_texts = outputs["documents"][i]
        
        for j in range(len(ranks)):
            answer_data = {
                "rank" : ranks[j],
                "chunk_name" : chunk_names[j],
                "chunk_text" : chunk_texts[j]
            }
            answer_chunks.append(answer_data)
        answers = {
            "question_name" : question_names[i],
            "question" : question,
            "returned_answers" : answer_chunks
        }
        qna.append(answers)
        
    return qna


def write_to_csv_top_5(path, new=False, results=None, model_name=None):
    if new:
        with open(path, "w") as f:
            writer = csv.writer(f, delimiter=",", )
            writer.writerow(["MODEL NAME", "TOP 1", "TOP 3", "TOP 5"])
    with open(path, "a") as f:
        writer = csv.writer(f, delimiter=",")
        results = list(results)
        results.insert(0, model_name)
        writer.writerow(results)
        print(f"Results successfully recorded. Model name: {model_name} Results: {results[1:]}")
