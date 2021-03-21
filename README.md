# bert_question_answer_NLI

# Objective
Use Google's BERT to classify question-answer dataset passage pairs by recognizing the answers with high lexical overlap. In other words, determine whether the context sentence contains the answer to the question.

The code is based on https://www.tensorflow.org/tutorials/text/solve_glue_tasks_using_bert_on_tpu and was carried out on SQuAD 2.0 (https://rajpurkar.github.io/SQuAD-explorer/) dataset.

This code attempts to re-create the task with the QNA dataset (https://microsoft.github.io/msmarco/#qna).


# Data Source
The data was sampled down to 60,000 samples due to computation limitations. The csv file containing these datapoints are in folder data_qna. The rest of the data can be downloaded from using the code below:

```
!wget 'https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz'
!wget 'https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz'
!wget 'https://msmarco.blob.core.windows.net/msmarco/eval_v2.1_public.json.gz'
!gzip -d train_v2.1.json.gz
!gzip -d dev_v2.1.json.gz
!gzip -d eval_v2.1_public.json.gz
```

