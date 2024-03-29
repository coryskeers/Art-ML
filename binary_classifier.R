# file input
file1 <- "/Users/cskeers/Documents/pil/picasso.txt"
file2 <- "/Users/cskeers/Documents/pil/warhol.txt"
artist_data1 <- read.table(file1, header=FALSE, sep=",", fill=TRUE)
artist_data2 <- read.table(file2, header=FALSE, sep=",", fill=TRUE)

# strip columns, add category
category1 = 0
category2 = 1
artist_data1 <- artist_data1[,c(-1,-12)]
artist_data1[,11] = category1
artist_data2 <- artist_data2[,c(-1,-12)]
artist_data2[,11] = category2

# create full dataset
df <- rbind(artist_data1, artist_data2)


# create train/test indices
num_rows = dim(df)[1]
num_train = as.integer(num_rows * 0.75)
train_index = sample(1:num_rows, num_train, replace = FALSE)
num_rows
num_train
test_index = setdiff(1:num_rows, train_index)
y_train = df[train_index,11]
y_test = df[test_index,11]
x_train = scale(df[train_index,-11])
x_test = scale(df[test_index,-11])

train = cbind.data.frame(y = y_train, x = x_train)
test = cbind.data.frame(y = y_test, x = x_test)

# create and test model
model = glm(y ~ ., data = train, family = "binomial")
summary(model)
fit = as.numeric(predict(model, newdata = test) > 0.5)
con = table(fit, test$y)
err1 = con[1,2] + con[2,1]
erate = err1 / sum(con)
con
erate
