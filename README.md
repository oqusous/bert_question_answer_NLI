# bert_question_answer_NLI

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


Labels for \[passages]\[0 to 808731]\[passages]\[is_selected] are 0 for bad answers and 1 for good answers. The data is imbalanced with following count or each class:
|label| count
|-----|--------|
|0    |7,536,988 |
|1    |532,761 |

The data was sampled down to 60,000 samples with 1:1 ratio of label 0 and 1 due to computational limitation and to resolved the data imbalance issue. The folder data_qna in this repo contains a csv file with the choosen sample. The dataset's first 5 rows is given below:

## Question and Answer Word Frequence Plots

Out of interest I wanted to view the top 150 words the occur in the questions and answers.

![Answers word frequency](/plots/freqDist_gw_answers_v0.png)

![Questions word frequency](/plots/freqDist_gw_questions_v0.png)

# BERT Model Architecture



# Results
|Metric | 60,000 Stratified Sample|
|-------|------------|
|Accuracy_score| 0.75 |
|f1_score| 0.70 |
|recall_score| 0.74 |
|precision_score| 0.66 |

A plot of probability of 'good answer' vs 'bad answer' for each for each datapoint that was successfuly and unsuccessfuly classified by the model.

![Questions word frequency](/plots/scatter_kde_hist_plot_v0.png)
