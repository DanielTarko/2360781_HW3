r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=64,
        seq_len=30,
        h_dim=128,
        n_layers=2,
        dropout=0.2,
        learn_rate=0.001,
        lr_sched_factor=0.5,
        lr_sched_patience=5,
    )
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    start_seq = "Once upon a time"
    temperature = 0.8
    return start_seq, temperature


part1_q1 = r"""
Splitting the corpus into sequences instead of training on the whole text has several advantages:

Memory Efficiency: Training on the entire text at once would require a large amount of memory, which may not be feasible. By splitting the text into smaller sequences, we can process and train on manageable chunks of data.

Batch Processing: Splitting the text into sequences allows us to create batches of data, which can be processed in parallel. This improves the efficiency of training and makes better use of computational resources.

Model Generalization: Training on smaller sequences helps the model learn to generalize better. It forces the model to learn patterns and dependencies within smaller contexts, which can then be applied to the entire text.

Handling Long-Term Dependencies: Many models, especially RNNs, struggle with very long sequences due to issues like vanishing gradients. By splitting the text into shorter sequences, we can mitigate this problem and help the model learn more effectively.

Incremental Learning: Splitting the text into sequences allows the model to learn incrementally. The model can update its parameters after processing each sequence, which can lead to faster convergence and better performance.

Overall, splitting the corpus into sequences makes the training process more efficient, manageable, and effective.



"""

part1_q2 = r"""
The hidden state allows the text to show memory longer than the sequence length. 

RNNs maintain a hidden state that is propagated through each time step of the input sequence. This hidden state acts as a memory that captures information from previous time steps. When generating text, the hidden state is carried forward from one sequence to the next, allowing the model to retain information beyond the length of a single sequence.

"""

part1_q3 = r"""
When training sequence models like RNNs, LSTMs, or GRUs, we often do not shuffle the order of batches to preserve the temporal dependencies within the sequences. Here are the main reasons:

Temporal Dependencies: Sequence models rely on the order of data to learn temporal dependencies. Shuffling the order of batches would disrupt these dependencies, making it difficult for the model to learn meaningful patterns over time.

Hidden State Continuity: In many sequence models, the hidden state is carried over from one batch to the next. If batches are shuffled, the continuity of the hidden state would be broken, leading to poor learning and performance.

Context Preservation: For tasks like language modeling or time series prediction, the context provided by previous data points is crucial. Shuffling batches would result in a loss of context, making it harder for the model to generate coherent and contextually accurate outputs.

"""

part1_q4 = r"""
1. Lowering the temperature during sampling makes the model's predictions more confident and less random. The temperature parameter controls the randomness of the predictions by scaling the logits before applying the softmax function. When the temperature is lower than 1.0, the logits are divided by a value greater than 1, which sharpens the probability distribution, making the model more likely to choose the most probable next character. This can lead to more coherent and sensible text generation.

2. When the temperature is very high (greater than 1.0), the logits are divided by a value less than 1, which flattens the probability distribution. This means that the differences between the probabilities of different characters are reduced, making the model more likely to sample from a wider range of characters, including less probable ones. As a result, the generated text becomes more random and less coherent, as the model is less confident in its predictions and more likely to make unpredictable choices.

3. When the temperature is very low (close to 0), the logits are divided by a very large value, which makes the probability distribution extremely sharp. In this case, the model becomes highly confident in its predictions and almost always chooses the character with the highest probability. This can lead to repetitive and deterministic text generation, as the model is less likely to explore alternative characters and more likely to stick to the most probable ones. This can result in less diverse and more predictable text.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL =  "https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip"



def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    #hypers["betas"] = #(a,b)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**

"""

# Part 3 answers
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=32,      # Moderate batch size for stable training
        h_dim=128,         # Hidden dimension for network capacity
        z_dim=100,         # Latent space dimension
        x_sigma2=0.01,     # Noise variance for added stability
        learn_rate=0.0002, # Learning rate (ADAM default for GANs)
        betas=(0.5, 0.999),# ADAM momentum parameters
        discriminator_optimizer=dict(
            type='Adam',
            lr=0.0002,
            betas=(0.5, 0.999)
        ),
        generator_optimizer=dict(
            type='Adam',
            lr=0.0002,
            betas=(0.5, 0.999)
        ),
        data_label=1.0,     # Label for real data
        label_noise=0.1     # Noise added to labels
    )
    return hypers

part3_q1 = r"""

In GAN training, whether gradients are maintained or discarded depends on which part of the network is being updated:

**Updating the Generator - Gradients Maintained:**

Gradients are kept because the generator's parameters are updated based on feedback from the discriminator.

The generator outputs fake samples, which the discriminator evaluates. Backpropagation flows through the discriminator and into the generator to adjust its parameters to create more realistic outputs.


**Updating the Discriminator - Gradients Discarded:**

Gradients are discarded for the generator because only the discriminator's parameters are updated in this step.

Fake samples from the generator are treated as static inputs, allowing the discriminator to learn to distinguish between real and fake data without affecting the generator.

This distinction optimizes training by retaining gradients only when they are necessary for parameter updates.

"""

part3_q2 = r"""
**Question 1**

No, Generator loss alone is not a reliable sole indicator for stopping GAN training because:

1. Discriminator loss is also important for training stability
2. The goal is to reach a Nash equilibrium, not just minimize generator loss. A consistently decreasing generator loss might actually indicate a weak discriminator that's easily fooled.
3. Generator might learn to fool discriminator without producing meaningful images

**Question 2**

Constant Discriminator Loss with Decreasing Generator Loss May Suggest:

1. The discriminator has reached a point where it struggles to distinguish between real and fake data.
2. Generator has learned to generate very similar/repetitive samples
3. Discriminator becomes "saturated" and can no longer effectively distinguish real from fake
4. Training instability or lack of diversity in generated samples

"""

part3_q3 = r"""
**Your answer:**



"""

PART3_CUSTOM_DATA_URL = "https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip"



def part4_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 128,  # Embedding dimension 
        num_heads = 4,    # Number of attention heads
        num_layers = 2,   # Number of transformer encoder layers
        hidden_dim = 256, # Hidden dimension in feed-forward network
        window_size = 128,# Sequence length 
        droupout = 0.1,   # Dropout rate
        lr = 1e-4,        # Learning rate
    )
    return hypers




part4_q1 = r"""

The stacking of encoder layers with sliding-window attention progressively expands the effective receptive field, similar to how convolutional neural networks (CNNs) work:

1. In the first layer, each token attends only to a local window of neighboring tokens.

2. In subsequent layers, the receptive field grows exponentially. This happens because:

    Each layer's output becomes the input for the next layer.

    Each token now represents information from its original local window, plus the local windows of neighboring tokens from the previous layer


3. By layer n, a token can indirectly gather information from a much broader context - approximately (window size)^n tokens away from its original position.

This is analogous to how CNN layers with small kernels can capture increasingly complex and wider-scale features as you stack more layers, enabling the network to understand broader contextual relationships without the computational expense of global attention mechanisms.


"""

part4_q2 = r"""

One possible variation of the attention pattern that maintains a similar computational complexity to the sliding-window attention (O(nw)), but allows for a more global context, is the dilated sliding-window attention. This approach introduces dilation into the sliding-window mechanism, allowing the model to attend to tokens that are further apart while keeping the computational complexity manageable.

**Dilated Sliding-Window Attention**

In dilated sliding-window attention, the window is applied with gaps (dilations) between the tokens it attends to. This allows each token to attend to a broader range of tokens without significantly increasing the number of tokens attended to within each window.

**How It Works**

1. Define Dilation: Introduce a dilation factor (d) that determines the spacing between the tokens within the window.
2. Apply Window with Dilation: Instead of attending to consecutive tokens within the window, attend to every (d)-th token within the window size.

**Benefits**
1. Broader Context: By introducing dilation, each token can attend to a more global context, capturing information from tokens that are further apart.
2. Controlled Complexity: The computational complexity remains (O(nw)) because the number of tokens attended to within each window remains constant, but the effective receptive field is larger.


"""


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
