# BERT_question_answer_NLI

# Objective
Use Google's BERT to classify question-answer dataset passage pairs by recognizing the answers with high lexical overlap. In other words, determine whether the context sentence contains the answer to the question.

The code is based on https://www.tensorflow.org/tutorials/text/solve_glue_tasks_using_bert_on_tpu and was carried out on SQuAD 2.0 (https://rajpurkar.github.io/SQuAD-explorer/) dataset.

This code attempts to re-create the task with the QNA dataset (https://microsoft.github.io/msmarco/#qna).

Google Colab is used as this is a memory intensive task which is best carried using GPU.

# Data Source
The dataset can be downloaded using the code below:

```
!wget 'https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz'
!wget 'https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz'
!wget 'https://msmarco.blob.core.windows.net/msmarco/eval_v2.1_public.json.gz'
!gzip -d train_v2.1.json.gz
!gzip -d dev_v2.1.json.gz
!gzip -d eval_v2.1_public.json.gz
```

The training set contains 8,069,749 answers to 808,731 questions. 

## Data Structure

Data is stored in a json file. A datapoint has the following structure:

```
dict_keys(['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'])
```
'answers' key:
```
 {
     '115868': ['It is a break in the upper quarter of the femur bone.'],
     '115869': ['No Answer Present.'],
     '11587': ['Penetrating the skin and increasing circulation to help rid the '
               'body of harmful toxins. Near and mid infrared enhance the benefits ',
      ...
 }
```
'query' key:
```
 {
    '115868': 'is reliability consistency',
    '115869': 'hip fractures',
    '11587': 'benefits of infrared sauna to skin',
    ...
 }
```
'wellFormedAnswers' key:
```
 {
    '115868': [],
    '115869': [],
    '11587':  [],
    ...
 }
```
'passages' key:
```
  '11587': [
            {'is_selected': 0,
              'passage_text': 'Provided a physician deems you healthy enough to take one, '
                              'saunas are very valuable for your skin. According to the '
                              'website Steam-Sauna-Benefits.com, enjoying a sauna can '
                              'relieve tension and stress and strengthen the immune '
                              'system. Beyond those values, Steam-Sauna-Benefits.com lauds '
                              'saunas for their ability to bolster the appearance of skin: '
                              'Taking a sauna can improve circulation, better the '
                              'reproduction of collagen and deeply cleanse and rejuvenate '
                              'your skin. Video of the Day',
              'url': 'https://www.livestrong.com/article/160396-benefits-of-sauna-for-skin/'},
             {'is_selected': 0,
              'passage_text': 'Traditional wet and dry saunas use heated air to warm the '
                              'body, which means they typically have to be uncomfortably '
                              'hot to reach therapeutic levels. Infrared saunas, on the '
                              'other hand, penetrate into tissues directly, causing the '
                              'body to sweat at a more comfortable ambient temperature. '
                              'Sauna Benefits. But why is it so important to sweat?',
              'url': 'https://www.mommypotamus.com/sauna-benefits/'},
              ... 
             {'is_selected': 1,
              'passage_text': 'It is the far infrared energy that is most beneficial, '
                              'penetrating the skin and increasing circulation to help rid '
                              'the body of harmful toxins. Near and mid infrared enhance '
                              'the benefits of far infrared.',
              'url': 'https://infraredsauna.com/infrared-sauna-health-benefits/'}
          ]
```


For the question-answering natural language inferencing task, the passages’ “is_selected” and “passage_text”, query and query_id parameters were to be kept for use in the training algorithm and subsequent testing of the model. Each entry has on average 10 passages as potential answers. When the data was re-structured into a tabular form that is more pandas/csv friendly, the query and query_id were repeated for each passage text and label- the resulting data is 7,536,988 passages labelled ‘0’ (not selected) and 532,761 labelled ‘1’ (selected).

|label| count
|-----|--------|
|0    |7,536,988 |
|1    |532,761 |

The dataset is huge and is highly unbalanced, due to time and computational limitations, the data had to be sampled down to a reasonable size. This is also created the opportunity of balancing the data through under-sampling. The data was sampled down to 30,000 unique queries, and for each query a passage labeled ‘0’ and a passage labeled ‘1’ were randomly selected, resulting in a balanced 60,000 datapoints.The folder data_qna_60k in this repo contains a csv file with the chosen sample. The dataset's first 5 rows is given below:

|query_id |	question |	sentence |	label |
|---------|----------|----------|-------|
|1036482 |	who is neyo ex wife?? |	But since she is now starring on “Atlanta Exes... |	0 |	
|119953 |	define de son tort	De Son Tort Law and Legal Definition. |	De Son T... |	1 |	
|1007602 |	which continent is covered in ice |	About 98% of Antarctica is covered by the Anta... |	1 |	
|17089 |	amount limits in rtgs and neft |	RTGS transactions involve large amounts of cas... |	1 |	
|569541 |	what are the elements of effective revision |	Organization is the progression, relatedness, ... |	0 |	

## Question and Answer Word Frequency Plots

Out of interest I wanted to view the top 150 words the occur in the questions and answers.

![Answers word frequency](/plots/freqDist_gw_answers_v0.png)

![Questions word frequency](/plots/freqDist_gw_questions_v0.png)

# BERT Model Architecture
The base BERT pre-processing and encoder pre-trained models were used for this task.

'bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'

The data is first tokenized to word ids using sentencepiece tokenizer. The preprocessing model then uses the ".bert_pack_inputs(tensors, seq_length)", which takes the list of tokens and a sequence length argument (128 - 256 is reasonable to use for the computing power provided by Google Colab). This packs the inputs to create a dictionary of tensors in the format expected by the BERT model.

![tokenizer](/plots/tok.PNG)

The output of the packer contains three inputs:

* input_mask: the mask allows the model to cleanly differentiate between the content and the padding. The mask has the same shape as the input_word_ids, and contains a 1 anywhere the input_word_ids is not padding.
* input_type_ids: it has the same shape of input_mask, but inside the non-padded region, contains a 0 or a 1 indicating which sentence the token is a part of.
* input_word_ids is the tokens with fixed sequence length, padding and the special 101 and 102 in the middle and end of the array tokens that indicate the start of the first sentence (question), start of the second sentence (answer) and start of the sequence padding. An example of what this may look like for one question-answer input pair is shown in the array below:
```
array([  101,  2054,  2003,  4786,  3255,  1999,  2026,  2398,   102,
       29267,  2389,  5234,  8715,  1012, 29267,  2389,  5234,  8715,
        2003,  1037,  9145,  4650,  3303,  2011,  1037, 18521,  9113,
        1999,  1996,  7223,  2008,  5260,  2000, 23229,  3255,  1010,
       11251,  1010,  2030, 15903,  2791,  1999,  1996,  5340,  2217,
        1997,  1996,  2192,  1010,  7223,  2030,  3093,  1012,  3949,
        2950,  3424,  1011, 20187,  4200,  1010, 11867,  4115,  3436,
        1996,  7223,  1010,  2030,  2522, 28228, 13186,  3334,  9314,
       13341,  2015,  1012,   102,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0], dtype=int32)
```

The BERT encoder layer is preceded by the input layer which receives the data pre-processed in the previous step. It is then followed by a dropout layer to control overfitting and final Dense classifier layer with Pureline activation function.

![Model](/plots/model.PNG)


# Results
|Metric | 60,000 Stratified Sample|
|-------|------------|
|Accuracy_score| 0.75 |
|f1_score| 0.70 |
|recall_score| 0.74 |
|precision_score| 0.66 |

A plot of Pureline activation output for 'good answer' vs 'bad answer' for each for each datapoint that was successfully and unsuccessfully classified by the model.

![Probability Scatter Plots](/plots/scatter_kde_hist_plot_v0.png)
