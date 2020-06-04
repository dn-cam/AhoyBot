# AhoyBot : A Chatbot that speaks Pirate

AhoyBot is a chatbot written in Python, using the computational power of Se2Seq models ([here's a helpful guide](https://towardsdatascience.com/day-1-2-attention-seq2seq-models-65df3f49e263#:~:text=A%20Seq2Seq%20model%20is%20a,outputs%20another%20sequence%20of%20items.&text=The%20encoder%20captures%20the%20context,then%20produces%20the%20output%20sequence.))AhoyBot was trained on the [Cornell Movie Dialog dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), after translating the entire movie corpus on using a Pirate language conversion dictionary.  

The goal behind this project was to find out of Seq2Seq models can learn to speak in particular fashion, i.e. have some sort of "personality" in the way it responds. Another important aspect of this project is to be able to maintain conversational context beyond more than just the present question asked.  

The Seq2Seq model used in this project uses an encoder-decoder pair, each consiting of 2 LSTM-RNN layers, along with Luong Attention Mechanism to preserve context of a conversation. The model was implmented using PyTorch library in Python. After training the model for 300,000 iterations we observed that the Chatbot gained considerable "Personality", but it still struggled in forming grammatically correct sentences. The latter was an expected result, since Seq2Seq models being of Generative nature, take a long time to "learn" the grammar of a language.

![GitHub Logo](https://github.com/dn-cam/AhoyBot/blob/master/Conversation%20Screenshots/Conversation_2.png)

## Running AhoyBot on your own machine

We wanted to host AhoyBot on a public url, but the bulky nature of a Pytorch installation on a webserver and also the big size of the models prevented us from doing so on the limited-size webhosts available free of cost. But it's easy to get AhoyBot running on your local system. Here are the instructions:  

1. Clone this git repo
2. Make sure Python 3.x is installed on your system, along with Python libraries PyTorch, Flask and Re.
3. run `python flask_app.py`
4. If you get any "module not found" error, please install those Python libraries, else, the app should be up and running on localhost, on port 5001
5. Open a browser of your choice, and hit the following url: `localhost:5001`
6. Now you should be able to converse with AhoyBot.

