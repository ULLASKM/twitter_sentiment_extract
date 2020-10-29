# Twitter Sentiment Extraction using Custom RobertaQA Transformer Model

### Problem Statement: Extract support phrases for sentiment labels

With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds. But, which words actually lead to the sentiment description? In this competition you will need to pick out the part of the tweet (word or phrase) that reflects the sentiment.

I would like to thank Chris Deotte for his wonderful kernel .This gave me intutional knowledge on roberta model.

### Model Approach :

a) Pre-trained-Model: (TFRobertaQA model was not yet released)

1. We use a pretrained-Roberta-base-model and upon that we add a custom question answer head layer (TFRobertaQA model was not yet released).
2. First tokens are input into bert_model and we use BERT's first output, i.e. x[0] below. These are embeddings of all input tokens and have shape (batch_size, MAX_LEN, 768).
3. Next we apply tf.keras.layers.Conv1D(filters=1, kernel_size=1) and transform the embeddings into shape (batch_size, MAX_LEN, 1).
4. We then flatten this and apply softmax, so our final output from x1 has shape (batch_size, MAX_LEN). These are one hot encodings of the start tokens indicies (for selected_text). And x2 are the end tokens indicies.

b) Loading Pre-trained model weights :
By using 5 folds Pretrained model weights and with leakyRelu layer on it we predict start and end indices of selected_text.
