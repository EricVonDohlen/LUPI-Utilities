# LUPI Simulation for Linear Logistic Regression




library(MASS)
library(caret)

set.seed(123)

# ----- 1. Simulate Data -----
n <- 500
X <- matrix(rnorm(n * 2), ncol = 2)
Xp <- matrix(rnorm(n * 2), ncol = 2)
beta_X  <- c(1.0, -1.2)
beta_Xp <- c(0.8, 0.5)
logits <- X %*% beta_X + Xp %*% beta_Xp
prob <- 1 / (1 + exp(-logits))
y <- rbinom(n, 1, prob)

data <- data.frame(y = y, X1 = X[,1], X2 = X[,2], Xp1 = Xp[,1], Xp2 = Xp[,2])

# ----- 2. Train Teacher Model (with privileged info) -----
teacher_model <- glm(y ~ X1 + X2 + Xp1 + Xp2, data = data, family = "binomial")
soft_labels <- predict(teacher_model, type = "response")

# ----- 3. Cross-Validation Setup -----
folds <- createFolds(y, k = 5, list = TRUE)
alpha_values <- seq(0, 1, by = 0.1)
cv_results <- data.frame(alpha = alpha_values, accuracy = NA)

# ----- 4. Cross-Validation Loop -----
for (a in alpha_values) {
  acc_list <- c()
  
  for (fold in folds) {
    train_idx <- setdiff(1:n, fold)
    test_idx <- fold
    
    # Blend targets
    blended_y <- a * y[train_idx] + (1 - a) * soft_labels[train_idx]
    train_df <- data.frame(y_blend = blended_y,
                           X1 = X[train_idx,1],
                           X2 = X[train_idx,2])
    
    test_df <- data.frame(X1 = X[test_idx,1],
                          X2 = X[test_idx,2])
    
    # Train student model
    student_model <- glm(y_blend ~ X1 + X2, data = train_df, family = "binomial")
    
    # Predict classes on test fold
    preds <- predict(student_model, newdata = test_df, type = "response")
    pred_classes <- ifelse(preds > 0.5, 1, 0)
    
    acc <- mean(pred_classes == y[test_idx])
    acc_list <- c(acc_list, acc)
  }
  cv_results$accuracy[cv_results$alpha == a] <- mean(acc_list)
}

# ----- 5. Find Best Alpha -----
best_alpha <- cv_results$alpha[which.max(cv_results$accuracy)]
best_accuracy <- max(cv_results$accuracy)

cat("Best alpha:", best_alpha, "\n")
cat("Cross-validated Accuracy:", round(best_accuracy, 3), "\n")

# Plot results
plot(cv_results$alpha, cv_results$accuracy, type="b", pch=19,
     xlab="Alpha (blending weight)",
     ylab="Cross-validated Accuracy",
     main="Optimizing Alpha for LUPI Logistic Regression")
abline(v = best_alpha, col = "red", lty = 2)
