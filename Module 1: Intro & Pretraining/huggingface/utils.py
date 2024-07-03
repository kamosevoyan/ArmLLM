from datasets import DatasetDict


def add_postfix(string: str, summarization_pad: str):
    return f"{string}\n{summarization_pad}"

def remove_postfix(string: str, summarization_pad: str):
    return string.split(summarization_pad)[-1]

def append_few_shot(document: str, dataset: DatasetDict):
    
    icl_subset = dataset['train'].shuffle().select(range(3))        
    prompt = []
    
    for item in icl_subset:
        document_ = item["document"]
        summary = item["summary"]
        
        postfixed_document = add_postfix(document_)
        shot_example = f"Text: {postfixed_document}{summary}"
        prompt.append(shot_example)
        
    prompt = "\n\n".join(prompt)
    prompt = f"{prompt}\n\nText: {add_postfix(document)}"
    
    return prompt