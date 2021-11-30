# AI Practice - Sentiment Analysis, PyTorch, Convnets, Regression

Disclaimer: This repo will probably turn out to be very messy. That's okay though as I want it to demonstrate my personal progress, along with the failures I encounter.  

## So, what is the goal of this code?
The goal of this repository is for me to learn fundamentals of AI. To do this, I want to create a neural network, that can predict how Elon Musk's Tweets affect the Bitcoin price.   
To better focus my learning, I have split the problem into 2 parts:
- A text sentiment neural network can can classify the overall mood of a given text (positive/negative)
- A regression model that will take in the tweet & sentiment, and output the predicted BTC price

I am currently working on the first one (text sentiment).

## My learnings so far:
Since I do not have access to the Twitter API yet, I used a dataset from IMDB movie reviews - which classifies the mood of the text as either positive or negative.

### First attempt at creating a Text-Sentiment classification model:
My first attempt was to assign a unique integer ID to every word in the dataset & then feed the words to the model in the order they were written in the review.

This attempt failed, because the model **did not learn**. I am assuming these reasons:
- Movie Reviews are of different lengths.
  - I used a custom size (input size of the net). If the text was too long, i simply cut it off. If the text was too short, I padded its end with "0" inputs
- The word indices were assigned randomly
  - I created a vocabulary file where each word got a random number
  - Those numbers did get very big
    - An attempt to fix this issue was to normalize them (add a calculation to make every input a number between -1 and 1), but it didn't help

### Discourse: Learning PyTorch with Convnets
While trying to create the Text-Sentiment model, I noticed that many of the issues I was facing were due to a weak understanding of PyTorch.
To deal with this problem I followed multiple tutorial series on the framework, and ended up creating a convolutional neural network that can classify images of cats and dogs.
Pretty straight-forward example, but it did teach me a lot about the framework, and also introduced me to convolution neural networks, which might be needed at a later stage of this project.


### Second attempt at creating the Text-Sentiment classification model:
This attempt is still in progress, and I am hoping to finish the text classification with the second try.
Here's my strategy:
 - Like the first time I have created a vocabluary (list of words with a unique ID for every single value)
   - I imporved the logic behind the generation of the vocabulary, making error inputs less likely
 - This time however, I intend to structure the model differently.  
   - Every input represents a word (so the input layer size = vocabulary size). Then I count how often a specific word is present in the movie review and map that value to the corresponding input neuron.
   - **Why do I this this will solve the issue?**
     - It makes the actual numbers given to the neural network much smaller
     - It creates a structure 
       - now, every neuron represents a word, unlike the previous attempt where each neuron represented a different word for every test case
       - The higher the value of the neuron, the more the word was used, which should give it more "power" when the model makes its classification. 
       - For example, the network will learn very fast which words (represented as neuron at position x) are meaningless to the sentiment (filler words such as "I", "and", "do", ..) and which are highly important (such as "fantastic", "bad", or "remarkable"). This information should help the network amke a accurate prediction.


 