# BTC1889H – Deep Learning
# Winter Term 2026
# Team Project
# Team 1: Viroosh Brendra, Agata Wolochacz, Lauren MacIntyre, Yaline Ganesathasan, Burrak Urrehman


# ============================================================
# SCRIPT SETUP
# ============================================================

# Clear environment 
rm(list = ls(all.names = TRUE))

# Load packages
library(keras3)
library(stringr)
library(tidyr)
library(ggplot2)

# Set random seed (ensures consistent results)
set.seed(123)
library(tensorflow)
tf$random$set_seed(123)

# Set working directory to current folder opened (if needed, update to your working directory)
setwd(".")


# ============================================================
# DATA PREPARATION AND PREPROCESSING 
# ============================================================

# ── 1. Load Raw Data ─────────────────────────────────────────
# df_train = training set | df_test = testing set
df_train <- read.csv("Corona_NLP_train.csv", stringsAsFactors = FALSE)
df_test  <- read.csv("Corona_NLP_test.csv",  stringsAsFactors = FALSE)

cat("Training samples:", nrow(df_train), "\n")
cat("Testing  samples:", nrow(df_test),  "\n")

cat("\nTraining sentiment distribution:\n")
print(table(df_train$Sentiment))

cat("\nTesting sentiment distribution:\n")
print(table(df_test$Sentiment))


# ── 2. Extract Raw Text and Labels ───────────────────────────
train_texts  <- df_train$OriginalTweet
test_texts   <- df_test$OriginalTweet
train_labels <- df_train$Sentiment
test_labels  <- df_test$Sentiment

# Tweets contain non-UTF-8 bytes (e.g. latin-1 special characters,
# emoji byte sequences) so converted each string to UTF-8, replacing any invalid bytes
# with a blank, then strip any remaining non-ASCII characters.
clean_encoding <- function(x) {
  x <- iconv(x, from = "", to = "UTF-8", sub = " ")   # replace bad bytes
  x <- iconv(x, from = "UTF-8", to = "ASCII", sub = " ") # drop non-ASCII
  x <- gsub("\\s+", " ", x)                            # collapse whitespace
  trimws(x)
}

train_texts <- clean_encoding(train_texts)
test_texts  <- clean_encoding(test_texts)


# ── 3. Explore Text Length (informs maxlen choice) ───────────
# Count words per tweet (approx, splitting on whitespace)
ll <- str_count(train_texts, "\\S+")

cat("\nTweet word-count summary (training set):\n")
print(summary(ll))
hist(ll, main = "Tweet word-count distribution (training)",
     xlab = "Number of words", col = "steelblue")

# What % of tweets fall within our chosen maxlen?3
# Following tutorial 9 approach of checking coverage
maxlen <- 50   # tweets are short; 50 covers the large majority
cat(sprintf("\nProportion of tweets with <= %d words: %.1f%%\n",
            maxlen, mean(ll <= maxlen) * 100))


# ── 4. Tokenization + Vectorization ──────────────────────────
# Following Tutorial 9: use layer_text_vectorization() + adapt()
# max_tokens = vocabulary size cap (top N most frequent words)
# output_mode = "int"  -> produces integer token ID sequences
# output_sequence_length = maxlen -> handles padding/truncation on-graph

max_words <- 10000   # vocabulary size; reasonable for tweet corpus

vec <- layer_text_vectorization(
  max_tokens = max_words,
  standardize = "lower_and_strip_punctuation",  # lowercases + strips punctuation
  split = "whitespace",
  output_mode = "int",
  output_sequence_length = maxlen               # pads/truncates to fixed length
)

# Fit vocabulary on TRAINING text only (avoid data leakage)
adapt(vec, train_texts)

# Inspect vocabulary
vocab <- vec$get_vocabulary()
cat("\nVocabulary size (including padding + OOV tokens):", length(vocab), "\n")
cat("Top 10 tokens:", paste(vocab[1:10], collapse = ", "), "\n")
# Note: index 0 = padding, index 1 = [UNK] (OOV), real words start at index 2


# ── 5. Produce Integer Sequences ─────────────────────────────
# Apply the fitted vectorizer to both splits
# Output is already padded/truncated to maxlen (as in Tutorial 9)
x_train <- as.array(vec(train_texts))   # shape: (n_train, maxlen)
x_test  <- as.array(vec(test_texts))    # shape: (n_test,  maxlen)

cat("\nShape of x_train:", paste(dim(x_train), collapse = " x "), "\n")
cat("Shape of x_test: ", paste(dim(x_test),  collapse = " x "), "\n")

# Example: first training tweet as integer sequence
cat("\nFirst training tweet (integer sequence):\n")
print(x_train[1, ])


# ── 6. Shuffle Training Data ─────────────────────────────────
# Following Tutorial 10: shuffle before training
set.seed(123)
I <- sample.int(nrow(x_train))
x_train      <- x_train[I, ]
train_labels <- train_labels[I]


# ── 7. Label Encoding ────────────────────────────────────────
# Five ordered sentiment classes -> integers 0-4 -> one-hot
# Ordinal order preserved: Extremely Negative (0) -> Extremely Positive (4)

sentiment_levels <- c(
  "Extremely Negative",   # 0
  "Negative",             # 1
  "Neutral",              # 2
  "Positive",             # 3
  "Extremely Positive"    # 4
)

NUM_CLASSES <- length(sentiment_levels)

# Integer encoding (0-indexed for Keras)
y_train_int <- match(train_labels, sentiment_levels) - 1L
y_test_int  <- match(test_labels,  sentiment_levels) - 1L

# No NAs means all labels were recognised
stopifnot(!any(is.na(y_train_int)), !any(is.na(y_test_int)))

# One-hot encode for categorical_crossentropy
y_train <- to_categorical(y_train_int, num_classes = NUM_CLASSES)
y_test  <- to_categorical(y_test_int,  num_classes = NUM_CLASSES)

cat("\nShape of y_train:", paste(dim(y_train), collapse = " x "), "\n")
cat("Shape of y_test: ", paste(dim(y_test),  collapse = " x "), "\n")

cat("\nLabel encoding:\n")
for (i in seq_along(sentiment_levels)) {
  cat(sprintf("  %d  ->  %s\n", i - 1L, sentiment_levels[i]))
}


# ── 8. Key Parameter Summary ─────────────────────────────────
cat("\n--- Preprocessing Parameters ---\n")
cat("max_words (vocabulary size) :", max_words,    "\n")
cat("maxlen (sequence length)    :", maxlen,       "\n")
cat("NUM_CLASSES                 :", NUM_CLASSES,  "\n")
cat("Training samples            :", nrow(x_train),"\n")
cat("Testing  samples            :", nrow(x_test), "\n")



# ============================================================
# MODEL DEVELOPMENT AND TRAINING
# Shared inputs: x_train, x_test, y_train, y_test
# ============================================================

# NOTE:
# All models below use the SAME preprocessing pipeline and training settings to ensure a fair comparison across architectures.


# 1. Shared Training Parameters
embedding_dim <- 50        # size of word embeddings (dense vector representation)
batch_size <- 32           # number of samples per training step
epochs <- 5                # number of full passes through the training data
validation_split <- 0.2    # proportion of training data used for validation

num_classes <- 5           # number of sentiment categories (multi-class output)

optimizer_type <- "adam"                     # optimization algorithm for training
loss_function <- "categorical_crossentropy"  # loss for multi-class classification
metrics_list <- c("accuracy")                # evaluation metric tracked during training


# 2. Supporting Functions for Model Training
compile_model <- function(model) {
  model %>% compile(
    optimizer = optimizer_type,   # applies same optimizer to all models
    loss = loss_function,         # ensures consistent loss function
    metrics = metrics_list        # tracks accuracy during training
  )
}

train_model <- function(model) {
  model %>% fit(
    x_train, y_train,                    # training data (same for all models)
    epochs = epochs,                     # number of training passes
    batch_size = batch_size,             # consistent batch size
    validation_split = validation_split, # monitors validation performance
    verbose = 1                          # shows training progress clearly
  )
}


# ============================================================
# FEEDFORWARD NEURAL NETWORK (FF)
# ============================================================

#1.FF model parameters selected
ff_units <- 64 #number of neurons in each hidden dense layer - captures meaningful patterns without making model very large
dropout_rate <- 0.2 # 20% of neurons silenced randomly during training

#2. build model function
build_ff_model <- function(two_layers = FALSE, use_dropout = FALSE) {
  
  model <- keras_model_sequential() %>%
    
#EMBEDDING LAYER = converts each integer word ID into a dense vector of numbers
#input_dim = vocabulary size, output_dim = vector size per word
#embedding_dim = 50 is appropriate for FF because it balances expressiveness and input size 
#after flattening, this produces 50 x 50 = 2500 inputs to the dense layer = large enough to capture word-level patterns
#without making the dense layer very large
#output shape: (batch, maxlen, embedding_dim) = (batch, 50, 50)
    layer_embedding(
      input_dim    = max_words,
      output_dim   = embedding_dim,
      input_length = maxlen
    ) %>%
    
#FLATTEN LAYER = unrolls the 2D embedding grid (50 words x 50 numbers) into
#one flat vector of 2500 numbers — required before dense layers
#output shape: (batch, 2500)
    layer_flatten()
  
  if (!two_layers) {   #case: single hidden dense layer
    
    if (!use_dropout) {
      model <- model %>%
        # single dense layer w/ no regularization
        # relu activation: output = max(0, input) for introducing non linearity
        layer_dense(units = ff_units, activation = "relu")
    } else {
      model <- model %>%
        layer_dense(units = ff_units, activation = "relu") %>%
        layer_dropout(rate = dropout_rate) #add dropout 
    }
    
  } else {   # case: two hidden dense layers
    
    if (!use_dropout) {
      model <- model %>%
        layer_dense(units = ff_units, activation = "relu") %>%
        # second dense layer learns combinations of the first layer's patterns
        layer_dense(units = ff_units, activation = "relu")
    } else {
      model <- model %>%
        layer_dense(units = ff_units, activation = "relu") %>%
        layer_dropout(rate = dropout_rate) %>% #adding dropout 
        layer_dense(units = ff_units, activation = "relu") %>%
        layer_dropout(rate = dropout_rate)
    }
  }
  
  model <- model %>%
    #OUTPUT LAYER: 5 units, one per sentiment class
    #softmax converts raw scores into probabilities that sum to 1
    #predicted class = whichever of the 5 has the highest probability
    layer_dense(units = num_classes, activation = "softmax")
  
  return(model)
}

#3.train the FF models 
# 3.1 single dense layer w/ no dropout = simplest baseline
ff_1layer <- build_ff_model(two_layers = FALSE, use_dropout = FALSE)
compile_model(ff_1layer)                    #adam optimizer, categorical crossentropy loss
history_ff_1layer <- train_model(ff_1layer) #5 epochs, batch 32, 20% validation split

# 3.2 single dense layer w/ dropout 
ff_1layer_dropout <- build_ff_model(two_layers = FALSE, use_dropout = TRUE)
compile_model(ff_1layer_dropout)
history_ff_1layer_dropout <- train_model(ff_1layer_dropout)

# 3.3 two dense layers w/ no dropout
ff_2layer <- build_ff_model(two_layers = TRUE, use_dropout = FALSE)
compile_model(ff_2layer)
history_ff_2layer <- train_model(ff_2layer)

# 3.4 two dense layers w/ dropout 
ff_2layer_dropout <- build_ff_model(two_layers = TRUE, use_dropout = TRUE)
compile_model(ff_2layer_dropout)
history_ff_2layer_dropout <- train_model(ff_2layer_dropout)

#4. evaluate FF models - run each trained model on the held-out test data (no weight updates here)
eval_ff_1layer <- evaluate(ff_1layer, x_test, y_test, verbose = 0)
eval_ff_1layer_dropout <- evaluate(ff_1layer_dropout, x_test, y_test, verbose = 0)
eval_ff_2layer <- evaluate(ff_2layer,x_test, y_test, verbose = 0)
eval_ff_2layer_dropout <- evaluate(ff_2layer_dropout, x_test, y_test, verbose = 0)

#plot training and validation accuracy+loss across epochs 
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 0, 15))

colors <- c("blue", "red", "green", "purple")

histories <- list(
  history_ff_1layer,
  history_ff_1layer_dropout,
  history_ff_2layer,
  history_ff_2layer_dropout
)

#ACCURACY
plot(NULL, xlim = c(1,5), ylim = c(0.3, 1.0),
     xlab = "Epoch", ylab = "Accuracy",
     main = "FF Models: Accuracy", xaxt = "n")
axis(1, at = 1:5)
for (i in 1:4) {
  lines(1:5, histories[[i]]$metrics$accuracy,     col = colors[i], lwd = 2, lty = 1)
  lines(1:5, histories[[i]]$metrics$val_accuracy, col = colors[i], lwd = 2, lty = 2)
}

#LOSS
plot(NULL, xlim = c(1,5), ylim = c(0.5, 2.5),
     xlab = "Epoch", ylab = "Loss",
     main = "FF Models: Loss", xaxt = "n")
axis(1, at = 1:5)
for (i in 1:4) {
  lines(1:5, histories[[i]]$metrics$loss,     col = colors[i], lwd = 2, lty = 1)
  lines(1:5, histories[[i]]$metrics$val_loss, col = colors[i], lwd = 2, lty = 2)
}

#add legend to right margin
par(xpd = NA)
legend(x      = par("usr")[2] + 1,
       y      = par("usr")[4],
       legend = c("1 Layer", "1 Layer + Dropout",
                  "2 Layers", "2 Layers + Dropout",
                  "Training", "Validation"),
       col    = c(colors, "black", "black"),
       lty    = c(1, 1, 1, 1, 1, 2),
       lwd    = 2,
       bty    = "n",
       cex    = 0.85)

par(mfrow = c(1, 1), xpd = FALSE) #reset


#5. summarize test results
ff_results <- data.frame(
  Model = c(
    "FF_1Layer",
    "FF_1Layer_Dropout",
    "FF_2Layer",
    "FF_2Layer_Dropout"
  ),
  Test_Loss = c(
    as.numeric(eval_ff_1layer["loss"]),
    as.numeric(eval_ff_1layer_dropout["loss"]),
    as.numeric(eval_ff_2layer["loss"]),
    as.numeric(eval_ff_2layer_dropout["loss"])
  ),
  Test_Accuracy = c(
    as.numeric(eval_ff_1layer["accuracy"]),
    as.numeric(eval_ff_1layer_dropout["accuracy"]),
    as.numeric(eval_ff_2layer["accuracy"]),
    as.numeric(eval_ff_2layer_dropout["accuracy"])
  )
)

print(ff_results)   

# ============================================================
# RECURRENT NEURAL NETWORK (RNN)
# ============================================================

# Model variants: single vs stacked simple RNN layers, with and without dropout.


# 1. RNN-Specific Parameters
rnn_units <- 32      # hidden units per RNN layer (consistent with LSTM for fair comparison)
dropout_rate <- 0.2  # dropout rate applied to input and recurrent connections


# 2. Build RNN Model Function
build_rnn_model <- function(two_layers = FALSE, use_dropout = FALSE) {
  
  model <- keras_model_sequential() %>%
    
    # EMBEDDING LAYER: maps integer token IDs to dense vectors
    # input_dim = vocabulary size, output_dim = embedding dimension
    # output shape: (batch, maxlen, embedding_dim)
    layer_embedding(
      input_dim    = max_words,
      output_dim   = embedding_dim,
      input_length = maxlen
    )
  
  if (!two_layers) {   # case: single RNN layer
    
    if (!use_dropout) {
      model <- model %>%
        # simple RNN: processes sequence step by step, returns final output only
        # output shape: (batch, rnn_units)
        layer_simple_rnn(units = rnn_units)
    } else {
      model <- model %>%
        layer_simple_rnn(
          units              = rnn_units,
          dropout            = dropout_rate,             # drops input connections
          recurrent_dropout  = dropout_rate              # drops recurrent connections
        )
    }
    
  } else {   # case: two stacked RNN layers
    
    if (!use_dropout) {
      model <- model %>%
        # first RNN must return full sequence so second layer receives sequential input
        layer_simple_rnn(units = rnn_units, return_sequences = TRUE) %>%
        layer_simple_rnn(units = rnn_units)              # second layer returns final output only
    } else {
      model <- model %>%
        layer_simple_rnn(
          units             = rnn_units,
          return_sequences  = TRUE,
          dropout           = dropout_rate,
          recurrent_dropout = dropout_rate
        ) %>%
        layer_simple_rnn(
          units             = rnn_units,
          dropout           = dropout_rate,
          recurrent_dropout = dropout_rate
        )
    }
  }
  
  model <- model %>%
    # OUTPUT LAYER: 5 units (one per sentiment class)
    # softmax converts raw scores to probabilities summing to 1
    layer_dense(units = num_classes, activation = "softmax")
  
  return(model)
}


# 3. Train RNN Model Variants

# 3.1 Single RNN layer, no dropout - simplest RNN baseline
rnn_1layer <- build_rnn_model(two_layers = FALSE, use_dropout = FALSE)
compile_model(rnn_1layer)
history_rnn_1layer <- train_model(rnn_1layer)

# 3.2 Single RNN layer, with dropout
rnn_1layer_dropout <- build_rnn_model(two_layers = FALSE, use_dropout = TRUE)
compile_model(rnn_1layer_dropout)
history_rnn_1layer_dropout <- train_model(rnn_1layer_dropout)

# 3.3 Two stacked RNN layers, no dropout
rnn_2layer <- build_rnn_model(two_layers = TRUE, use_dropout = FALSE)
compile_model(rnn_2layer)
history_rnn_2layer <- train_model(rnn_2layer)

# 3.4 Two stacked RNN layers, with dropout
rnn_2layer_dropout <- build_rnn_model(two_layers = TRUE, use_dropout = TRUE)
compile_model(rnn_2layer_dropout)
history_rnn_2layer_dropout <- train_model(rnn_2layer_dropout)


# 3.5 Plot Training and Validation Accuracy + Loss
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 0, 15))

colors <- c("blue", "red", "green", "purple")

histories_rnn <- list(
  history_rnn_1layer,
  history_rnn_1layer_dropout,
  history_rnn_2layer,
  history_rnn_2layer_dropout
)

# ACCURACY
plot(NULL, xlim = c(1, 5), ylim = c(0.3, 1.0),
     xlab = "Epoch", ylab = "Accuracy",
     main = "RNN Models: Accuracy", xaxt = "n")
axis(1, at = 1:5)
for (i in 1:4) {
  lines(1:5, histories_rnn[[i]]$metrics$accuracy,     col = colors[i], lwd = 2, lty = 1)
  lines(1:5, histories_rnn[[i]]$metrics$val_accuracy, col = colors[i], lwd = 2, lty = 2)
}

# LOSS
plot(NULL, xlim = c(1, 5), ylim = c(0.5, 2.5),
     xlab = "Epoch", ylab = "Loss",
     main = "RNN Models: Loss", xaxt = "n")
axis(1, at = 1:5)
for (i in 1:4) {
  lines(1:5, histories_rnn[[i]]$metrics$loss,     col = colors[i], lwd = 2, lty = 1)
  lines(1:5, histories_rnn[[i]]$metrics$val_loss, col = colors[i], lwd = 2, lty = 2)
}

# Legend in right margin
par(xpd = NA)
legend(x      = par("usr")[2] + 1,
       y      = par("usr")[4],
       legend = c("1 Layer", "1 Layer + Dropout",
                  "2 Layers", "2 Layers + Dropout",
                  "Training", "Validation"),
       col    = c(colors, "black", "black"),
       lty    = c(1, 1, 1, 1, 1, 2),
       lwd    = 2,
       bty    = "n",
       cex    = 0.85)

par(mfrow = c(1, 1), xpd = FALSE)   # reset


# 4. Evaluate RNN Models on Test Data
eval_rnn_1layer         <- evaluate(rnn_1layer,         x_test, y_test, verbose = 0)
eval_rnn_1layer_dropout <- evaluate(rnn_1layer_dropout, x_test, y_test, verbose = 0)
eval_rnn_2layer         <- evaluate(rnn_2layer,         x_test, y_test, verbose = 0)
eval_rnn_2layer_dropout <- evaluate(rnn_2layer_dropout, x_test, y_test, verbose = 0)


# 5. Summarize Test Results
rnn_results <- data.frame(
  Model = c(
    "RNN_1Layer",
    "RNN_1Layer_Dropout",
    "RNN_2Layer",
    "RNN_2Layer_Dropout"
  ),
  Test_Loss = c(
    as.numeric(eval_rnn_1layer["loss"]),
    as.numeric(eval_rnn_1layer_dropout["loss"]),
    as.numeric(eval_rnn_2layer["loss"]),
    as.numeric(eval_rnn_2layer_dropout["loss"])
  ),
  Test_Accuracy = c(
    as.numeric(eval_rnn_1layer["accuracy"]),
    as.numeric(eval_rnn_1layer_dropout["accuracy"]),
    as.numeric(eval_rnn_2layer["accuracy"]),
    as.numeric(eval_rnn_2layer_dropout["accuracy"])
  )
)

print(rnn_results)

# ============================================================
# LONG SHORT-TERM MEMORY (LSTM) — Lauren
# ============================================================

# LSTM models are designed to capture sequential dependencies in text.
# They improve upon basic RNNs by retaining important information over longer sequences.


# 1. LSTM-Specific Parameters
lstm_units <- 32      # number of hidden units (controls model capacity)
dropout_rate <- 0.2   # proportion of connections randomly dropped during training (regularization)


# 2. Build LSTM Model Function
build_lstm_model <- function(two_layers = FALSE, use_dropout = FALSE) {
  
  model <- keras_model_sequential() %>%   # initialize sequential neural network
    
    layer_embedding(
      input_dim = max_words,         # size of vocabulary (from preprocessing)
      output_dim = embedding_dim,    # dimension of learned word vectors
      input_length = maxlen          # fixed length of each input sequence
    )
  
  if (!two_layers) {   # case: single LSTM layer
    
    if (!use_dropout) {
      model <- model %>%
        layer_lstm(units = lstm_units)   # standard LSTM layer (returns final output only)
    } else {
      model <- model %>%
        layer_lstm(
          units = lstm_units,
          dropout = dropout_rate,             # drops input connections
          recurrent_dropout = dropout_rate    # drops recurrent (memory) connections
        )
    }
    
  } else {   # case: two stacked LSTM layers
    
    if (!use_dropout) {
      model <- model %>%
        layer_lstm(
          units = lstm_units,
          return_sequences = TRUE            # required to pass full sequence to next LSTM
        ) %>%
        layer_lstm(units = lstm_units)       # second LSTM processes sequence output
    } else {
      model <- model %>%
        layer_lstm(
          units = lstm_units,
          return_sequences = TRUE,
          dropout = dropout_rate,
          recurrent_dropout = dropout_rate
        ) %>%
        layer_lstm(
          units = lstm_units,
          dropout = dropout_rate,
          recurrent_dropout = dropout_rate
        )
    }
  }
  
  model <- model %>%
    layer_dense(
      units = num_classes,             # number of output classes (5 sentiments)
      activation = "softmax"           # outputs probability distribution over classes
    )
  
  return(model)                        # return completed model architecture
}


# 3. Train LSTM Model Variants

# 3.1 Single LSTM layer, no dropout
lstm_1layer <- build_lstm_model(two_layers = FALSE, use_dropout = FALSE)  # baseline LSTM
compile_model(lstm_1layer)                                               # apply shared compile settings
history_lstm_1layer <- train_model(lstm_1layer)                           # train and store training history

# 3.2 Single LSTM layer, with dropout
lstm_1layer_dropout <- build_lstm_model(two_layers = FALSE, use_dropout = TRUE)  # adds regularization
compile_model(lstm_1layer_dropout)
history_lstm_1layer_dropout <- train_model(lstm_1layer_dropout)

# 3.3 Two LSTM layers, no dropout
lstm_2layer <- build_lstm_model(two_layers = TRUE, use_dropout = FALSE)  # deeper model (more complexity)
compile_model(lstm_2layer)
history_lstm_2layer <- train_model(lstm_2layer)

# 3.4 Two LSTM layers, with dropout
lstm_2layer_dropout <- build_lstm_model(two_layers = TRUE, use_dropout = TRUE)  # deeper + regularized
compile_model(lstm_2layer_dropout)
history_lstm_2layer_dropout <- train_model(lstm_2layer_dropout)

# 3.5 Plot Training Performance (Validation Accuracy)
# Combine validation accuracy across all LSTM model variants into one data frame
df_plot <- data.frame(
  epoch = 1:epochs,   # x-axis: training epochs
  
  LSTM_1Layer = history_lstm_1layer$metrics$val_accuracy,                 # baseline model
  LSTM_1Layer_Dropout = history_lstm_1layer_dropout$metrics$val_accuracy, # single layer + dropout
  LSTM_2Layer = history_lstm_2layer$metrics$val_accuracy,                 # deeper model (2 layers)
  LSTM_2Layer_Dropout = history_lstm_2layer_dropout$metrics$val_accuracy  # deeper + dropout
)
# Reshape data from wide → long format for ggplot compatibility
df_long <- pivot_longer(
  df_plot,
  -epoch,                              # keep epoch as identifier
  names_to = "Model",                  # model names become a column
  values_to = "Validation_Accuracy"    # corresponding values stored here
)
# Plot validation accuracy across epochs for all models
ggplot(df_long, aes(
  x = epoch,                           # x-axis = training epochs
  y = Validation_Accuracy,             # y-axis = validation accuracy
  color = Model,                       # different color per model
  linetype = Model                     # different line style for clarity
)) +
  geom_line(size = 1) +                # draw lines for each model
  # Manually assign colors for consistency and readability
  scale_color_manual(values = c(
    "LSTM_1Layer" = "purple",           # baseline model
    "LSTM_1Layer_Dropout" = "blue",    # dropout version (highlighted)
    "LSTM_2Layer" = "red",             # deeper model
    "LSTM_2Layer_Dropout" = "green" # deeper + dropout
  )) +
  labs(
    title = "Validation Accuracy Across Epochs for LSTM Models",  # figure title
    x = "Epoch",                                                  # x-axis label
    y = "Validation Accuracy"                                     # y-axis label
  ) +
  theme_minimal()   # clean, publication-style theme


# 4. Evaluate LSTM Models
# Evaluate performance on unseen test data (final model comparison)
eval_lstm_1layer <- evaluate(lstm_1layer, x_test, y_test, verbose = 0)
eval_lstm_1layer_dropout <- evaluate(lstm_1layer_dropout, x_test, y_test, verbose = 0)
eval_lstm_2layer <- evaluate(lstm_2layer, x_test, y_test, verbose = 0)
eval_lstm_2layer_dropout <- evaluate(lstm_2layer_dropout, x_test, y_test, verbose = 0)


# 5. Summarize Test Results
# Store results in a structured table for comparison across models
lstm_results <- data.frame(
  Model = c(
    "LSTM_1Layer",
    "LSTM_1Layer_Dropout",
    "LSTM_2Layer",
    "LSTM_2Layer_Dropout"
  ),
  Test_Loss = c(
    as.numeric(eval_lstm_1layer["loss"]),            # model error on test data
    as.numeric(eval_lstm_1layer_dropout["loss"]),
    as.numeric(eval_lstm_2layer["loss"]),
    as.numeric(eval_lstm_2layer_dropout["loss"])
  ),
  Test_Accuracy = c(
    as.numeric(eval_lstm_1layer["accuracy"]),        # classification accuracy
    as.numeric(eval_lstm_1layer_dropout["accuracy"]),
    as.numeric(eval_lstm_2layer["accuracy"]),
    as.numeric(eval_lstm_2layer_dropout["accuracy"])
  )
)

print(lstm_results)   # display final comparison of all LSTM variants

# ============================================================
# MODEL EVALUATION AND COMPARISON
# ============================================================

# This section brings together results from all three architectures
# (FF, RNN, LSTM) and evaluates them using:
#   (1) standard test accuracy
#   (2) a custom ordinal MSE metric that accounts for the ordered
#       structure of the five sentiment classes


# ── 1. Master Comparison Table ───────────────────────────────
# Combine per-architecture result tables into one data frame

all_results <- rbind(ff_results, rnn_results, lstm_results)
all_results$Architecture <- c(rep("FF", 4), rep("RNN", 4), rep("LSTM", 4))
all_results <- all_results[, c("Architecture", "Model", "Test_Loss", "Test_Accuracy")]

cat("\n--- All Model Test Results ---\n")
print(all_results[order(-all_results$Test_Accuracy), ], row.names = FALSE, digits = 4)


# ── 2. Custom Ordinal Metric: Mean Squared Error ─────────────
# Custom ordinal metric: convert predicted probabilities to class labels and compute
# mean squared error between predicted and true ordinal classes.
compute_ordinal_mse <- function(model, x, y_int) {
  # Predict class probabilities, shape: (n_samples, 5)
  probs <- predict(model, x, verbose = 0)
  # Convert to 0-indexed predicted class (argmax)
  pred_class <- apply(probs, 1, which.max) - 1L
  # Return mean squared ordinal distance
  mean((pred_class - y_int)^2)
}

cat("\n--- Computing Ordinal MSE for all 12 model variants ---\n")

model_objects <- list(
  ff_1layer, ff_1layer_dropout, ff_2layer, ff_2layer_dropout,
  rnn_1layer, rnn_1layer_dropout, rnn_2layer, rnn_2layer_dropout,
  lstm_1layer, lstm_1layer_dropout, lstm_2layer, lstm_2layer_dropout
)

all_results$Ordinal_MSE <- sapply(model_objects, compute_ordinal_mse,
                                   x = x_test, y_int = y_test_int)

# Sort by accuracy for display
all_results_sorted <- all_results[order(-all_results$Test_Accuracy), ]

cat("\n--- Full Comparison Table (sorted by Test Accuracy) ---\n")
print(all_results_sorted[, c("Architecture","Model","Test_Loss","Test_Accuracy","Ordinal_MSE")],
      row.names = FALSE, digits = 4)


# ── 3. Best Model Per Architecture ───────────────────────────
# For each architecture, select the variant with the highest test accuracy

best_ff_row   <- all_results[all_results$Architecture == "FF",   ]
best_rnn_row  <- all_results[all_results$Architecture == "RNN",  ]
best_lstm_row <- all_results[all_results$Architecture == "LSTM", ]

best_per_arch <- rbind(
  best_ff_row[which.max(best_ff_row$Test_Accuracy),   ],
  best_rnn_row[which.max(best_rnn_row$Test_Accuracy), ],
  best_lstm_row[which.max(best_lstm_row$Test_Accuracy),]
)

cat("\n--- Best Model Per Architecture ---\n")
print(best_per_arch[, c("Architecture","Model","Test_Loss","Test_Accuracy","Ordinal_MSE")],
      row.names = FALSE, digits = 4)


# ── 4. Visualisations ────────────────────────────────────────

arch_colors <- c("FF" = "steelblue", "RNN" = "tomato", "LSTM" = "seagreen")

# 4.1 Bar chart: Test Accuracy for all 12 models, colour-coded by architecture
# Ordered by accuracy to make ranking immediately visible

all_results$Model_ordered <- factor(all_results$Model,
                                     levels = all_results$Model[order(all_results$Test_Accuracy)])

ggplot(all_results, aes(x = Model_ordered, y = Test_Accuracy, fill = Architecture)) +
  geom_bar(stat = "identity", width = 0.7) +
  scale_fill_manual(values = arch_colors) +
  coord_flip() +
  labs(
    title = "Test Accuracy: All Model Variants",
    x     = NULL,
    y     = "Test Accuracy"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")


# 4.2 Bar chart: Ordinal MSE for all 12 models (lower = better)
all_results$Model_ordered_mse <- factor(all_results$Model,
                                         levels = all_results$Model[order(-all_results$Ordinal_MSE)])

ggplot(all_results, aes(x = Model_ordered_mse, y = Ordinal_MSE, fill = Architecture)) +
  geom_bar(stat = "identity", width = 0.7) +
  scale_fill_manual(values = arch_colors) +
  coord_flip() +
  labs(
    title = "Ordinal MSE: All Model Variants (lower = better)",
    x     = NULL,
    y     = "Ordinal MSE"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")


# 4.3 Scatter: Accuracy vs. Ordinal MSE (ideal model sits top-left)
# Best models are in the upper-left corner: high accuracy + low MSE
ggplot(all_results, aes(x = Ordinal_MSE, y = Test_Accuracy,
                         color = Architecture, label = Model)) +
  geom_point(size = 3) +
  geom_text(vjust = -0.7, size = 3, check_overlap = TRUE) +
  scale_color_manual(values = arch_colors) +
  labs(
    title = "Accuracy vs. Ordinal MSE Across All Models",
    x     = "Ordinal MSE (lower = better)",
    y     = "Test Accuracy (higher = better)"
  ) +
  theme_minimal()


# ── 5. Confusion Matrix for the Best Overall Model ───────────

best_idx        <- which.max(all_results$Test_Accuracy)
best_model_name <- as.character(all_results$Model[best_idx])
best_arch       <- as.character(all_results$Architecture[best_idx])

cat(sprintf("\nBest overall model : %s  (%s)\n",  best_model_name, best_arch))
cat(sprintf("  Test Accuracy    : %.4f\n", all_results$Test_Accuracy[best_idx]))
cat(sprintf("  Test Loss        : %.4f\n", all_results$Test_Loss[best_idx]))
cat(sprintf("  Ordinal MSE      : %.4f\n", all_results$Ordinal_MSE[best_idx]))

# Retrieve the Keras model object that corresponds to the winner
best_model_obj <- switch(best_model_name,
  "FF_1Layer"           = ff_1layer,
  "FF_1Layer_Dropout"   = ff_1layer_dropout,
  "FF_2Layer"           = ff_2layer,
  "FF_2Layer_Dropout"   = ff_2layer_dropout,
  "RNN_1Layer"          = rnn_1layer,
  "RNN_1Layer_Dropout"  = rnn_1layer_dropout,
  "RNN_2Layer"          = rnn_2layer,
  "RNN_2Layer_Dropout"  = rnn_2layer_dropout,
  "LSTM_1Layer"         = lstm_1layer,
  "LSTM_1Layer_Dropout" = lstm_1layer_dropout,
  "LSTM_2Layer"         = lstm_2layer,
  "LSTM_2Layer_Dropout" = lstm_2layer_dropout
)

# Predict on test set and build confusion matrix
probs_best <- predict(best_model_obj, x_test, verbose = 0)
pred_best  <- apply(probs_best, 1, which.max) - 1L   # 0-indexed predicted class

# Confusion matrix: rows = true class, columns = predicted class
conf_mat <- table(
  True      = factor(sentiment_levels[y_test_int + 1], levels = sentiment_levels),
  Predicted = factor(sentiment_levels[pred_best  + 1], levels = sentiment_levels)
)

cat("\nConfusion Matrix (Best Overall Model):\n")
print(conf_mat)

# 5.1 Heatmap of the confusion matrix
conf_df <- as.data.frame(conf_mat)
conf_df$True      <- factor(conf_df$True,      levels = rev(sentiment_levels))
conf_df$Predicted <- factor(conf_df$Predicted, levels = sentiment_levels)

ggplot(conf_df, aes(x = Predicted, y = True, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = sprintf("Confusion Matrix — %s (%s)", best_model_name, best_arch),
    x     = "Predicted Sentiment",
    y     = "True Sentiment",
    fill  = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# Conclusions:
cat("\nBest model per architecture:\n")
print(best_per_arch[, c("Architecture", "Model", "Test_Accuracy", "Ordinal_MSE")],
      row.names = FALSE, digits = 4)

# Print best model
cat(sprintf(
  "\nOverall winner: %s (%s)\n",
  best_model_name, best_arch
))


