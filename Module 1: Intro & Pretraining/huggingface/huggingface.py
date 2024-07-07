import torch
from datasets import DatasetDict, load_dataset
from evaluate import load
from torch.nn import Module
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import add_postfix, append_few_shot, remove_postfix

# model choices
LLAMA_2_9B = "google/gemma-2-9b"
GEMMA_2_27B = "google/gemma-2-27b"
LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B"
LLAMA_3_70B = "meta-llama/Meta-Llama-3-70B"

SUMMARIZATION_PAD = "TL;DR: "


def simple_inference(model: Module, tokenizer: AutoTokenizer, dataset: DatasetDict):

    document = """Internet searches from the week before the crash were found on the tablet computer used by Andreas Lubitz, Meanwhile, the second "black box" flight recorder from the plane has been recovered. There were no survivors among the 150 people on board the A320 on 24 March. The German prosecutors said internet searches made on the tablet found in Lubitz's Duesseldorf flat included "ways to commit suicide" and "cockpit doors and their security provisions". Spokesman Ralf Herrenbrueck said: "He concerned himself on one hand with medical treatment methods, on the other hand with types and ways of going about a suicide. "In addition, on at least one day he concerned himself with search terms about cockpit doors and their security precautions.'' Prosecutors did not disclose the individual search terms in the browser history but said personal correspondence supported the conclusion Lubitz used the device in the period from 16 to 23 March. Lubitz, 27, had been deemed fit to fly by his employers at Germanwings, a subsidiary of Lufthansa. The first "black box", the voice recorder, was recovered almost immediately at the crash site. Based on that evidence, investigators said they believed Lubitz intentionally crashed Flight 9525, which was travelling from Barcelona to Duesseldorf, taking control of the aircraft while the pilot was locked out of the cockpit. The second "black box" recovered is the flight data recorder (FDR) which should hold technical information on the time of radio transmissions and the plane's acceleration, airspeed, altitude and direction, plus the use of auto-pilot. At a press conference, Marseille prosecutor Brice Robin said there was "reasonable hope" the recorder which was being sent to Paris for examination, would provide useful information. The "completely blackened" equipment was found near a ravine and was not discovered immediately because it was the same colour as the rocks, he said. He said: "The second black box is an indispensable addition to understand what happened especially in the final moment of the flight." He told the media 150 separate DNA profiles had been isolated from the crash site but he stressed that did not mean all the victims had been identified. As each DNA set is matched to a victim, families will be notified immediately, he said, He added 40 mobile phones had been recovered. He said they would be analysed in a laboratory but were "heavily damaged". Also on Thursday, Germanwings said it was unaware that Lubitz had experienced depression while he was training to be a pilot. Lufthansa confirmed on Tuesday that it knew six years ago that the co-pilot had suffered from an episode of "severe depression'' before he finished his flight training. ``We didn't know this,'' said Vanessa Torres, a spokeswoman for Lufthansa subsidiary Germanwings, which hired Lubitz in September 2013. She could not explain why Germanwings had not been informed. The final minutes Lubitz began the jet's descent at 10:31 (09:31 GMT) on 24 March, shortly after the A320 had made its final contact with air traffic control. Little more than eight minutes later, it had crashed into a mountain near Seyne-les-Alpes. What happened in the last 30 minutes of Flight 4U 9525? Who was Andreas Lubitz?"""
    # postfixed_document = add_postfix(document)
    postfixed_document = append_few_shot(document, dataset)

    encoded_document = tokenizer(postfixed_document, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = model.generate(**encoded_document, max_new_tokens=32)

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    prediction = remove_postfix(decoded_output)

    print(f"Prediction:\n{prediction}")


def eval_dataset(model: Module, tokenizer: AutoTokenizer, dataset: DatasetDict):

    subset = dataset["validation"].shuffle().select(range(200))
    metric = load("rouge")

    for data in tqdm(subset):

        document = data["document"]
        summary = data["summary"]

        # postfixed_document = add_postfix(document)
        postfixed_document = append_few_shot(document, dataset)

        encoded_docoment = tokenizer(postfixed_document, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                output = model.generate(
                    **encoded_docoment, max_new_tokens=32, do_sample=False
                )

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        prediction = remove_postfix(decoded_output)
        metric.add(prediction=prediction, reference=summary)

    metric_result = metric.compute()
    print(metric_result)


def eval_dataset_batch(
    model: Module, tokenizer: AutoTokenizer, dataset: DatasetDict, batch_size: int = 2
):

    subset = (
        dataset["validation"]
        .shuffle()
        .select(range(200))
        .map(lambda x: {"length": len(x["document"]), **x})
        .sort("length")
    )
    metric = load("rouge")

    for begin_idx in tqdm(range(0, len(subset), batch_size)):

        batch = subset[begin_idx : begin_idx + batch_size]
        batch_documents = batch["document"]
        batch_summaries = batch["summary"]

        postfixed_batch_documents = [
            # add_postfix(document)
            append_few_shot(document, dataset)
            for document in batch_documents
        ]

        postfixed_batch_documents = tokenizer(
            postfixed_batch_documents,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    **postfixed_batch_documents, max_new_tokens=32, do_sample=False
                )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_outputs = list(map(lambda x: remove_postfix(x), decoded_outputs))
        metric.add_batch(references=batch_summaries, predictions=decoded_outputs)

    metric_result = metric.compute()
    print(metric_result)


if __name__ == "__main__":

    model_name = GEMMA_2_27B

    # initialize thetokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # initialize quantization config
    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # initialize the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        # quantization_config=q_config,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # load the dataset
    dataset = load_dataset("EdinburghNLP/xsum")

    simple_inference(model=model, tokenizer=tokenizer, dataset=dataset)
    # eval_dataset(model=model, tokenizer=tokenizer, dataset=dataset)
    # eval_dataset_batch(model=model, tokenizer=tokenizer, dataset=dataset)
