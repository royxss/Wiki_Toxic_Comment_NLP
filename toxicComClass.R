setwd("C:\\Users\\SROY\\Documents\\CodeBase\\Datasets\\toxicCommentClassif")
rm(list=ls())
seedVal = 17869
#memory.limit(40000)

# Load libraries
library(dplyr)
library(NLP)
library(tm)
library(stringr)
library(textstem)
library(h2o)
library(lime)
library(xgboost)
library(text2vec)
library(caret)
library(pROC)

# Load saved file
load("TCC1.RData")

# Import Data
test <- read.csv2('test.csv', sep=',')
train <- read.csv2('train.csv', sep=',')
samplesub <- read.csv2('sample_submission.csv', sep=',')

# Convert into datatype
str(train)
train <- train %>% mutate_if(is.integer,funs(factor(.)))
train$comment_text <- as.character(train$comment_text)


contentTransformCleanup <-  function(x){
  x <- gsub(" ?(f|ht)tp(s?)://(.*)[.][a-z]+", " ", x)        # Remove URL
  x <- gsub(" ?(f|ht)tp(s?)://.*", " ", x)                   # Remove partial urls
  x <- gsub("?(@|#)\\w+", " ", x)                            # Remove hashtags
  x <- gsub("<.*?>", " ", x)                                 # Remove hrefs
  #x <- gsub("[[:punct:]]", " ", x)                          # Remove punctuations
  x <- gsub("[^[:alnum:]']", " ", x)                         # Remove punctuations except apostrophe
  x <- gsub("(\\s\\'|\\'\\s)", " ", x)                       # Remove single quotes from words; check apostrophe;;not robust
  x <- gsub("[[:digit:]]", " ", x)                           # Remove numbers
  #x <- gsub("\\b\\d+\\b", " ", x)                           # Remove only numbers and not like s80n etc
  #x <- gsub("\\b\\w{1,2}\\b[^']", " ", x)                   # DOESN'T WORK. Remove characters with 2 words or less; exclude apostrophe
  x <- gsub("\\s+", " ", x)                                  # Remove white spaces
  #x <- gsub("[^[:alnum:]]", " ", x)                         # Remove alphanumeric
  return(tolower(x))
}


# Use above function to clean data
parsedComment <- train %>% select(comment_text) %>% mutate_all(funs(contentTransformCleanup))
train$parsedComment <- parsedComment$comment_text


contentTransformStemLemma <-  function(x){
  #x <- stem_strings(x)               # Stemmer
  x <- lemmatize_strings(x)           # Lemmatizer
  x <- gsub("\\s+", " ", x)           # stem/lemma causes space issues
  return(x)
}

# Use above function to clean data
parsedComment <- train %>% select(parsedComment) %>% mutate_all(funs(contentTransformStemLemma))
train$parsedComment <- parsedComment$parsedComment


# More cleanup is required after lemmatization
# Weird characters present Ã¯Â¿Â½ Ã¯Â¿Â½. Also good to remove apostrophe at this point
STOP_WORDS <- stopwords("en")
contentTransformExtraCleanup <-  function(x){
  x <- gsub("[^0-9A-Za-z///' ]", " ", x)  # Remove weird characters
  x <- removeWords(x, STOP_WORDS)         # Remove stopwords
  x <- gsub("'", "", x)                   # Remove apostrophe
  x <- gsub("\\s+", " ", x)               # Remove extra space        
  return(x)
}

# Use above function to clean data
parsedComment <- train %>% select(parsedComment) %>% mutate_all(funs(contentTransformExtraCleanup))
train$parsedComment <- parsedComment$parsedComment

#saved 1
#save.image("TCC1.RData")

# Check class imbalance
n <- nrow(train)
(nrow(train[train$toxic == 0,])/n)
(nrow(train[train$severe_toxic == 0,])/n)
(nrow(train[train$obscene == 0,])/n)
(nrow(train[train$threat == 0,])/n)
(nrow(train[train$insult == 0,])/n)
(nrow(train[train$identity_hate == 0,])/n)


manual_words <- c('utc', tolower(month.name))
contentTransformManualCleanup <-  function(x){
  x <- gsub("\\b\\w{1,2}\\b", " ", x)     # remove single/double characters
  x <- gsub("\\b\\w{20,}\\b", " ", x)     # remove word length >= 20;....12 worked best
  x <- removeWords(x, manual_words)       # Remove words manually
  x <- gsub("\\s+", " ", x)               # Remove extra space        
  return(x)
}

# Use above function to clean data
parsedComment <- train %>% select(parsedComment) %>% mutate_all(funs(contentTransformManualCleanup))
train$parsedComment <- parsedComment$parsedComment

################################ simple model using xgb ######################

trainPct <- .8
testPct <- 1 - trainPct
inTrain <- createDataPartition(y = train$toxic, p = trainPct, list = FALSE)

it_train = itoken(train[inTrain,'parsedComment'], 
                  tokenizer = word_tokenizer, 
                  ids = train[inTrain,'id'], 
                  progressbar = FALSE)
vocab = create_vocabulary(it_train)
vocab = prune_vocabulary(vocab, term_count_min = 20L)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)
tfidf = TfIdf$new()
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
dtm_train_tfidf_l1_norm = normalize(dtm_train_tfidf, "l1")

# Check collocation
model2 = Collocations$new(vocabulary = vocab, collocation_count_min = 50, pmi_min = 0)
model2$partial_fit(it_train)
coll.df <- as.data.frame(model2$collocation_stat[pmi >= 8 & gensim >= 10 & lfmd >= -25, ])

it_test = itoken(train[-inTrain,'parsedComment'], 
                  tokenizer = word_tokenizer, 
                  ids = train[-inTrain,'id'], 
                  progressbar = FALSE)
dtm_test_tfidf  = create_dtm(it_test, vectorizer) %>% transform(tfidf) # Notice uses same vocab of train
dtm_test_tfidf_l1_norm = normalize(dtm_test_tfidf, "l1")

# Create boosting model for binary classification (-> logistic loss)
# Other parameters are quite standard
param <- list(eta = 0.1 
              ,objective = "binary:logistic" 
              ,eval_metric = "error" 
              ,nthread = 1
              ,max_delta_step = 10) #Class imbalance

xgb_model <- xgb.train(
  param, 
  xgb.DMatrix(dtm_train_tfidf_l1_norm, label = as.integer(as.character(train[inTrain,'toxic']))),
  nrounds = 50
)

# We use a (standard) threshold of 0.5
predictions <- ifelse(predict(xgb_model, dtm_test_tfidf_l1_norm) > 0.5, 1, 0)
test_labels <- as.integer(as.character(train[-inTrain,'toxic']))

# Accuracy
print(mean(predictions == test_labels))
confusion <- confusionMatrix(predictions, test_labels)
confusion
roc_obj <- roc(predictions, test_labels)
auc(roc_obj)

# LIME

# We select 10 sentences from the label
tt <- train[-inTrain,]
sentence_to_explain <- head(tt[test_labels == 1,]$parsedComment, 5)
explainer <- lime(sentence_to_explain, model = xgb_model, 
                  preprocess = get_matrix)
explanation <- explain(sentence_to_explain, explainer, n_labels = 1, 
                       n_features = 2)

# Most of the words choosen by Lime
# are related to the team (we, our)
# or part of the paper (Section, in)
expl <- explanation[, 2:9]
expl <- expl[expl$label_prob < 0.6 & expl$label_prob > 0.4,]

plot_features(explanation)

plot_text_explanations(explanation)

################################# h2o ##################################
# Start h2o
h2o.init(nthreads=-1, max_mem_size="4g")
#h2o.startLogging() 

tokenize <- function(sentences) {
  tokenized <- h2o.tokenize(as.character(as.h2o(sentences)),"") # Check space
  
  # remove short words (less than 3 characters)
  tokenized.lengths <- h2o.nchar(tokenized)
  #tokenized.filtered <- tokenized[is.na(tokenized.lengths) || tokenized.lengths > 2,]
  return(tokenized.lengths)
}

print("Break job titles into sequence of words")
words <- tokenize(train$parsedComment)

print("Build word2vec model")
# Test it for dry run
epoch = 1
vectors = 100
w2v.model <- h2o.word2vec(words
                          #, model_id = "w2v_model"
                          , vec_size = vectors
                          #, min_word_freq = 4
                          #, window_size = 5
                          #, init_learning_rate = 0.03
                          , sent_sample_rate = 0
                          , epochs = epoch)
h2o.performance(w2v.model)

#save model
#save.image("TCC2.RData")
#h2o.saveModel("Saveh2oW2V"
#              , path = "C:\\Users\\SROY\\Documents\\CodeBase\\Datasets\\toxicCommentClassif"
#              , force=TRUE)

#print("Sanity check - find synonyms for the word 'discuss'")
#print(h2o.findSynonyms(w2v.model, "discuss", count = 5))

print("Calculate a vector for each job title")
parsedComment.vec <- h2o.transform(w2v.model, words, aggregate_method = "AVERAGE")

print("Prepare training&validation data (keep only job titles made of known words)")
data <- h2o.cbind(as.h2o(train[, "toxic"]), parsedComment.vec)
data.split <- h2o.splitFrame(data, ratios = 0.8, seed=seedVal)

print("Build a basic GBM model")
gbm.model <- h2o.gbm(x = names(parsedComment.vec), y = "x",
                     training_frame = data.split[[1]], validation_frame = data.split[[2]],
                     seed=seedVal,
                     balance_classes = TRUE)
h2o.performance(gbm.model)

print("Predict!")
predict <- function(sentences, w2v, gbm) {
  words <- tokenize(sentences)
  parsedComment.vec <- h2o.transform(w2v, words, aggregate_method = "AVERAGE")
  h2o.predict(gbm, parsedComment.vec)
}

predict("die you bitch", w2v.model, gbm.model)

h2o.shutdown(prompt = FALSE)
