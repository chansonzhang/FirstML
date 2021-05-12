data(Nile)
(out <- bu.test(Nile))
par(mfrow=c(2,1))
start=1871
cp=unname(out$estimate)
x=start+cp-1
plot(Nile)
abline(v=x,col='red')
plot(out)
abline(v=x,col='red')

