sim_fit = function(n, p_old = 0.10, p_new = 0.25) {
  # create the data
  dead_old = rbinom(n, size = 1, prob = p_old)
  dead_new = rbinom(n, size = 1, prob = p_new)
  # create the predictor variable
  method = rep(c("old", "new"), each = n)
  # create a data.frame to pass to glm
  df = data.frame(dead = c(dead_old, dead_new), method = method)
  # relevel so old is the reference
  df$method = relevel(factor(df$method), ref = "old")
  # fit the model
  fit = glm(dead ~ method, data = df, family = binomial)
  # extract the p-value
  pval = summary(fit)$coef[2,4]
  # determine if it was found to be significant
  sig_pval = pval < 0.05
  # obtain the estimated mortality rate for the new method
  p_new_est = predict(fit, data.frame(method = c("new")),
                      type = "response")
  
  # determine if it is +/- 5% from the true value
  prc_est = p_new_est >= (p_new - 0.05) & p_new_est <= (p_new + 0.05)
  # return a vector with these two elements
  c(sig_pval = sig_pval, prc_est = unname(prc_est))
}

# containers: 
out_sig = matrix(NA, I, N) # matrix with I rows and N columns
out_prc = matrix(NA, I, N) # matrix with I rows and N columns
for (n in 1:N) {
  for (i in 1:I) {
    tmp = sim_fit(n = n_try[n])     # run sim
    out_sig[i,n] = tmp["sig_pval"]  # extract and store significance metric
    out_prc[i,n] = tmp["prc_est"]   # extract and store precision metric
  }
}

par(mfrow = c(1,2), mar = c(4,4,1,0))
plot(apply(out_sig, 2, mean) ~ n_try, type = "l",
     xlab = "Tagged Fish per Treatment",
     ylab = "Probability of Finding Effect (Power)")
plot(apply(out_prc, 2, mean) ~ n_try, type = "l",
     xlab = "Tagged Fish per Treatment",
     ylab = "Probability of a Precise Estimate")

